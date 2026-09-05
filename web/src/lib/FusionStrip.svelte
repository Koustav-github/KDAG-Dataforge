<script>
  // The merge, drawn in its settled state.
  //
  // Under `concat` the two parents sit side by side and the axis is 2n long —
  // both parents survive intact, which is the architectural claim. Under
  // `average` B lands *on top of* A and the two blend into a single n-long
  // strip, which is the only fusion a Transformer could attempt and the reason
  // it destroys both parents.
  //
  // Bar heights are real: each is the L2 norm of that neuron's row in the
  // decoder, so the texture you see is the trained model's own structure, not
  // generated noise.
  import { onMount } from 'svelte';

  export let parentA = null;
  export let parentB = null;
  export let mergeMode = 'concat';
  export let theta = null;      // triggers a redraw when the selected θ changes
  export let shared = null;     // concepts the two vocabularies share, exact
  export let totalConcepts = 24;
  export let damage = null;     // live merge damage at this θ, or null

  // θ does NOT change concat's geometry — the same 2048 neurons are glued the
  // same way at every θ. That is the architectural point, not an oversight. So
  // rather than animate a fake θ-dependence, show the two things θ genuinely
  // does move: the vocabulary overlap (exact, from the lexicons) and the damage
  // the merge causes. The operation is θ-independent; the outcome is not.
  $: overlapFrac = shared === null ? 0 : shared / totalConcepts;

  const A_COLOR = [10, 143, 114];    // #0a8f72
  const B_COLOR = [194, 94, 18];     // #c25e12

  const BAR_H = 64;
  const LABEL_H = 20;
  const H = BAR_H + LABEL_H;
  const GAP = 0.10;                  // resting gap, as a fraction of one parent's width

  let canvasEl;
  let wrapEl;
  let cssW = 640;

  // Drawn in the fused state, not animated into it. Under concat the motion was
  // identical at every θ — it showed the two blocks sliding together and nothing
  // else, while costing a full 640-column redraw per frame, which is what made
  // it stutter. The fused layout itself carries the information: two intact
  // halves under concat, one blended strip under average.

  // Per-neuron magnitude from the decoder's rows: shape (n, D), row-major.
  function magnitudes(model) {
    if (!model) return null;
    const dec = model.tensors.decoder;
    const D = model.cfg.n_embd;
    const n = dec.shape[0];
    const out = new Float32Array(n);
    let max = 0;
    for (let j = 0; j < n; j++) {
      let acc = 0;
      const off = j * D;
      for (let i = 0; i < D; i++) acc += dec.data[off + i] * dec.data[off + i];
      const v = Math.sqrt(acc);
      out[j] = v;
      if (v > max) max = v;
    }
    if (max > 0) for (let j = 0; j < n; j++) out[j] /= max;
    return out;
  }

  $: magA = magnitudes(parentA);
  $: magB = magnitudes(parentB);
  $: n = parentA ? parentA.n : 0;

  function bars(ctx, mag, x0, widthPx, color, alpha) {
    if (!mag) return;
    const count = mag.length;
    const step = widthPx / count;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
    // Draw in device columns rather than one rect per neuron: n=1024 bars in
    // ~300px means several neurons share a column, so take the column max.
    const cols = Math.max(1, Math.round(widthPx));
    for (let c = 0; c < cols; c++) {
      const lo = Math.floor((c / cols) * count);
      const hi = Math.max(lo + 1, Math.floor(((c + 1) / cols) * count));
      let m = 0;
      for (let j = lo; j < hi && j < count; j++) if (mag[j] > m) m = mag[j];
      const h = 6 + m * (BAR_H - 14);
      ctx.fillRect(x0 + c, (BAR_H - h) / 2, 1, h);
    }
    ctx.globalAlpha = 1;
  }

  function draw() {
    if (!canvasEl || !magA || !magB) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cssW;
    canvasEl.width = Math.round(w * dpr);
    canvasEl.height = Math.round(H * dpr);
    const ctx = canvasEl.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);

    const concat = mergeMode === 'concat';
    // One parent's width: at rest, two parents plus a gap must fit the canvas.
    const unit = w / (2 + GAP);

    // Where each parent's block sits once fused.
    const ax = concat ? 0 : (w - unit) / 2;
    const bx = concat ? unit : (w - unit) / 2;      // average: B sits ON A

    bars(ctx, magA, ax, unit, A_COLOR, 1);
    // Under average B is drawn at half weight over A — the visual analogue
    // of (a + b) / 2.
    bars(ctx, magB, bx, unit, B_COLOR, concat ? 1 : 0.5);

    if (!concat) {
      // the muddy result of averaging the two
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = '#8b8476';
      ctx.fillRect(ax, (BAR_H - 10) / 2, unit, 10);
      ctx.globalAlpha = 1;
    }

    // Each block is labelled in place, so which half belongs to which parent
    // never has to be inferred from colour alone.
    ctx.font = '600 11px -apple-system, Segoe UI, Roboto, sans-serif';
    ctx.textAlign = 'center';
    const labelY = BAR_H + 14;
    if (concat) {
      ctx.fillStyle = `rgb(${A_COLOR[0]},${A_COLOR[1]},${A_COLOR[2]})`;
      ctx.fillText(`parent A · 0–${n - 1}`, ax + unit / 2, labelY);
      ctx.fillStyle = `rgb(${B_COLOR[0]},${B_COLOR[1]},${B_COLOR[2]})`;
      ctx.fillText(`parent B · ${n}–${2 * n - 1}`, bx + unit / 2, labelY);
    } else {
      ctx.fillStyle = '#6b6459';
      ctx.fillText(`A + B averaged · 0–${n - 1}`, ax + unit / 2, labelY);
    }

    // seam marker: where A's neurons end and B's begin
    if (concat) {
      ctx.strokeStyle = '#1c1b19';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(unit, 2);
      ctx.lineTo(unit, BAR_H - 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }


  function measure() {
    if (wrapEl) cssW = Math.max(240, wrapEl.clientWidth);
    draw();
  }

  onMount(() => {
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  });

  // Redraw whenever the merge or its inputs change.
  $: if (magA && magB && mergeMode !== undefined && theta !== undefined && canvasEl) measure();

</script>

<figure class="fusion" bind:this={wrapEl}>
  {#if shared !== null}
    <div class="overlap">
      <span class="ov-label">vocabulary overlap at θ = {theta ?? '—'}</span>
      <svg viewBox="0 0 320 34" role="img"
           aria-label={`Language A and language B each name ${totalConcepts} concepts and share ${shared} of them.`}>
        <rect x="6" y="8" width={150} height="18" rx="3" fill="#0a8f72" opacity="0.28" />
        <rect x={164 - overlapFrac * 150} y="8" width={150} height="18" rx="3"
              fill="#c25e12" opacity="0.28" />
        <text x="160" y="21" class="ov-n" text-anchor="middle">{shared}/{totalConcepts}</text>
      </svg>
      <span class="ov-note">
        {shared === 0
          ? 'nothing shared — every concept has a different word in each language'
          : shared === totalConcepts
            ? 'identical vocabularies'
            : `${shared} of ${totalConcepts} concepts share a word`}
      </span>
    </div>
  {/if}

  <div role="img"
       aria-label={`Fusion of two ${n}-neuron parents by ${mergeMode}. ${
         mergeMode === 'concat'
           ? `They come to rest side by side, giving a ${2 * n}-neuron axis with both parents intact.`
           : `They land on top of each other and blend into a single ${n}-neuron axis.`}`}>
    <canvas bind:this={canvasEl} style="height:{H}px" aria-hidden="true"></canvas>
  </div>

  <figcaption>
    {#if mergeMode === 'concat'}
      <strong>concat</strong> — the two axes come to rest end to end:
      <code>{n} + {n} → {2 * n}</code>. Both parents survive; the dashed seam is where A ends and B begins.
      <br />
      <span class="theta-note">
        This layout is the same at every θ, and that is the point: concatenation glues the same
        {2 * n} neurons the same way regardless of how much vocabulary the parents share. θ changes
        the <em>outcome</em>, not the operation —
        {#if damage}damage here is <strong>{(damage.dMean >= 0 ? '+' : '') + damage.dMean.toFixed(3)}</strong>,
        against {(0.93).toFixed(2)} at θ = 0 and {(0.46).toFixed(2)} at θ = 1.{:else}see the damage figure in Act 3.{/if}
      </span>
    {:else}
      <strong>average</strong> — the two land on the same {n} neurons and blend.
      Nothing grows, and neither parent survives intact. This is the only fusion a Transformer could attempt.
    {/if}
  </figcaption>
</figure>

<style>
  .overlap { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
  .overlap svg { width: 190px; height: auto; }
  .ov-label { font-size: 0.72rem; color: var(--muted); }
  .ov-note { font-size: 0.72rem; color: var(--muted); }
  .ov-n { font-size: 11px; font-weight: 700; fill: var(--fg); }
  .theta-note { display: inline-block; margin-top: 0.25rem; }
  .fusion { margin: 0; display: flex; flex-direction: column; gap: 0.35rem; }
  canvas {
    width: 100%;
    display: block;
    border: 1px solid var(--border);
    border-radius: 5px;
    background: var(--tok-bg);
  }
  figcaption {
    font-size: 0.74rem;
    color: var(--muted);
    line-height: 1.45;
  }
  code {
    font-family: ui-monospace, Consolas, monospace;
    font-size: 0.72rem;
    color: var(--fg);
  }
</style>
