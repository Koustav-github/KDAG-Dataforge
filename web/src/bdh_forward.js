// Recurrent BDH-GPU forward pass (Eq. 8, arXiv:2509.26507), ported from
// model/src/bdh_surgery/recurrent.py and checked against it by test_js_parity.py.

const TENSOR_NAMES = [
  'decoder', 'encoder', 'encoder_v', 'attn.freqs', 'lm_head', 'embed.weight',
];

function dequantize(int8, scale) {
  const out = new Float32Array(int8.length);
  for (let i = 0; i < int8.length; i++) out[i] = int8[i] * scale;
  return out;
}

export async function loadModel(manifest, theta, side, fetchBin) {
  const frag = manifest.featured[theta][side];
  const tensors = {};
  for (const name of TENSOR_NAMES) {
    const meta = frag[name];
    // int8 tensors are dequantized once, here, into Float32Array — so merge.js
    // and forward() below never see a quantized value and need no changes.
    const data = meta.dtype === 'int8'
      ? dequantize(await fetchBin(meta.file, 'int8'), meta.scale)
      : await fetchBin(meta.file);
    tensors[name] = { data, shape: meta.shape, concatAxis: meta.concat_axis };
  }
  return { cfg: manifest.config, n: manifest.config.n, tensors };
}

export async function loadModelFromDir(dir, theta, side) {
  const fs = await import('node:fs/promises');
  const manifest = JSON.parse(await fs.readFile(`${dir}/manifest.json`, 'utf8'));
  return loadModel(manifest, theta, side, async (file, kind) => {
    const buf = await fs.readFile(`${dir}/${file}`);
    return kind === 'int8'
      ? new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength)
      : new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  });
}

function layerNorm(v) {
  const d = v.length;
  let mean = 0; for (let i = 0; i < d; i++) mean += v[i]; mean /= d;
  let varr = 0; for (let i = 0; i < d; i++) { const t = v[i] - mean; varr += t * t; }
  const inv = 1 / Math.sqrt(varr / d + 1e-5);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = (v[i] - mean) * inv;
  return out;
}

// x (D) @ encoder (1, D, n) -> (n), then ReLU
function encodeRelu(x, enc, D, n) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (let i = 0; i < D; i++) acc += x[i] * enc[i * n + j];
    out[j] = acc > 0 ? acc : 0;
  }
  return out;
}

// RoPE over adjacent pairs on the n axis, matching Attention.rope
function rope(v, freqs, t, n) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j += 2) {
    const ph = ((t * freqs[j]) % 1) * 2 * Math.PI;
    const c = Math.cos(ph), s = Math.sin(ph);
    out[j] = v[j] * c - v[j + 1] * s;
    out[j + 1] = v[j + 1] * c + v[j] * s;
  }
  return out;
}

export function forward(model, tokenIds, { ablated = null } = {}) {
  const { n_layer: L, n_embd: D, vocab_size: V, n_head: nh } = model.cfg;
  if (nh !== 1) throw new Error(`forward() is single-head only; got n_head=${nh}`);
  const n = model.n;
  const T = model.tensors;
  const mask = new Float32Array(n).fill(1);
  if (ablated) for (const i of ablated) mask[i] = 0;

  const rho = [];
  for (let l = 0; l < L; l++) rho.push(new Float32Array(n * D));
  const outputs = [];

  for (let t = 0; t < tokenIds.length; t++) {
    let x = layerNorm(T['embed.weight'].data.slice(tokenIds[t] * D, tokenIds[t] * D + D));
    for (let l = 0; l < L; l++) {
      const xs = encodeRelu(x, T.encoder.data, D, n);
      for (let j = 0; j < n; j++) xs[j] *= mask[j];
      const qr = rope(xs, T['attn.freqs'].data, t, n);

      const yKV = new Float32Array(D);                       // read before write
      for (let j = 0; j < n; j++) {
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yKV[i] += q * rho[l][off + i];
      }
      for (let j = 0; j < n; j++) {                          // then write
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) rho[l][off + i] += q * x[i];
      }

      const ys = encodeRelu(layerNorm(yKV), T.encoder_v.data, D, n);
      const yMLP = new Float32Array(D);
      for (let j = 0; j < n; j++) {
        const xy = xs[j] * ys[j]; if (xy === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yMLP[i] += xy * T.decoder.data[off + i];
      }
      const y = layerNorm(yMLP);
      const sum = new Float32Array(D);
      for (let i = 0; i < D; i++) sum[i] = x[i] + y[i];
      x = layerNorm(sum);
    }
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let acc = 0;
      for (let i = 0; i < D; i++) acc += x[i] * T.lm_head.data[i * V + v];
      logits[v] = acc;
    }
    outputs.push(logits);
  }
  return outputs;
}

// Mean per-neuron `xy` (= x_sparse * y_sparse, the sparse MLP activation
// that feeds the decoder) across every layer and sequence position, for one
// token sequence. Mirrors model/src/bdh_surgery/overlap.py:activation_profiles,
// which averages over layers then over tokens — since every layer sees the
// same T positions, that two-step average equals one mean over layers*T,
// which is what this computes directly. A separate function so forward()'s
// signature and return value are untouched (bdh_forward.js is covered by
// test_js_parity.py / test_js_merge.py).
export function neuronActivations(model, tokenIds) {
  const { n_layer: L, n_embd: D, n_head: nh } = model.cfg;
  if (nh !== 1) throw new Error(`neuronActivations() is single-head only; got n_head=${nh}`);
  const n = model.n;
  const T = model.tensors;

  const rho = [];
  for (let l = 0; l < L; l++) rho.push(new Float32Array(n * D));
  const sums = new Float64Array(n);
  let count = 0;

  for (let t = 0; t < tokenIds.length; t++) {
    let x = layerNorm(T['embed.weight'].data.slice(tokenIds[t] * D, tokenIds[t] * D + D));
    for (let l = 0; l < L; l++) {
      const xs = encodeRelu(x, T.encoder.data, D, n);
      const qr = rope(xs, T['attn.freqs'].data, t, n);

      const yKV = new Float32Array(D);                       // read before write
      for (let j = 0; j < n; j++) {
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yKV[i] += q * rho[l][off + i];
      }
      for (let j = 0; j < n; j++) {                          // then write
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) rho[l][off + i] += q * x[i];
      }

      const ys = encodeRelu(layerNorm(yKV), T.encoder_v.data, D, n);
      const yMLP = new Float32Array(D);
      for (let j = 0; j < n; j++) {
        const xy = xs[j] * ys[j];
        sums[j] += xy;
        if (xy === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yMLP[i] += xy * T.decoder.data[off + i];
      }
      const y = layerNorm(yMLP);
      const sum = new Float32Array(D);
      for (let i = 0; i < D; i++) sum[i] = x[i] + y[i];
      x = layerNorm(sum);
      count++;
    }
  }

  const out = new Float32Array(n);
  const inv = count ? 1 / count : 0;
  for (let j = 0; j < n; j++) out[j] = sums[j] * inv;
  return out;
}
