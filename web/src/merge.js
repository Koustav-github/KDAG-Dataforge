// Client-side merge. Classification comes from the manifest's concat_axis,
// never re-derived here — that is what keeps it in step with merge.py.
function catAxis(a, b, shape, axis) {
  if (axis === shape.length - 1) {           // interleave along the last axis
    const inner = shape[axis], rows = a.length / inner;
    const out = new Float32Array(a.length + b.length);
    for (let r = 0; r < rows; r++) {
      out.set(a.subarray(r * inner, (r + 1) * inner), r * 2 * inner);
      out.set(b.subarray(r * inner, (r + 1) * inner), r * 2 * inner + inner);
    }
    return out;
  }
  const out = new Float32Array(a.length + b.length);  // axis 0: plain append
  out.set(a, 0); out.set(b, a.length);
  return out;
}

function average(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
  return out;
}

function blend(a, b, concat) {
  const tensors = {};
  for (const name of Object.keys(a.tensors)) {
    const ta = a.tensors[name], tb = b.tensors[name];
    const axis = ta.concatAxis;
    if (concat && axis !== null) {
      const shape = ta.shape.slice();
      shape[axis] *= 2;
      tensors[name] = { data: catAxis(ta.data, tb.data, ta.shape, axis),
                        shape, concatAxis: axis };
    } else {
      tensors[name] = { data: average(ta.data, tb.data),
                        shape: ta.shape.slice(), concatAxis: axis };
    }
  }
  const n = concat ? a.n * 2 : a.n;
  return { cfg: { ...a.cfg, n }, n, tensors };
}

export const mergeConcat = (a, b) => blend(a, b, true);
export const mergeAverage = (a, b) => blend(a, b, false);
