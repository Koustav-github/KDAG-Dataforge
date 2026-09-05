<script>
  // Act 4: surgery. Finds the neurons responsible for merge damage (the
  // "collision set" — neurons strongly and equally active on both parent
  // directions), ablates them, and compares against a size-matched random
  // control. Every number here is computed live via forward(); nothing is
  // precomputed offline.
  import { collisionScores, collisionBreakdown, topK, randomNeurons } from '../collision.js';
  import { forward } from '../bdh_forward.js';
  import { ablated } from './store.js';
  import { meanLossAsync } from './compute.js';
  import SurgeryDerivation from './SurgeryDerivation.svelte';
  import { onMount } from 'svelte';
  import AblationStrip from './AblationStrip.svelte';

  export let model = null;   // merged model: { cfg, n, tensors }
  export let probes = null;  // full probes.json: { pivot, eval_a, eval_b }

  // The random-ablation control is a single draw's worth of noise if we stop
  // at one seed: at the shipped default (theta=0.5, k=40) the headline
  // difference measures -0.021, and the spread across draws at that point is
  // SD ~= 0.011-0.018 (model/scripts/measure_locality.py) — i.e. within noise.
  // So the control here averages over several draws and reports the mean
  // *and* its spread, instead of a single fixed seed standing in for "random".
  const SEED = 7; // first of the draw seeds; also fixes the heatmap's tick marks
  const N_RANDOM_DRAWS = 5;   // 8 blocked the thread for ~3.5 s per k change
  const DRAW_SEEDS = Array.from({ length: N_RANDOM_DRAWS }, (_, i) => SEED + i);
  // The loss comparison recomputes on every k change, so it stays on a small
  // probe subset to stay fast; collisionScores (below) uses the full eval
  // sets and is cached, since it does not depend on k at all.
  const LOSS_ROWS_PER_SIDE = 3;

  let k = 40;

  function sameSet(a, b) {
    if (a.length !== b.length) return false;
    const sa = new Set(a);
    for (const v of b) if (!sa.has(v)) return false;
    return true;
  }

  // Cached: this block's dependencies are `model` and `probes` only, so it
  // does NOT re-run when `k` changes — moving the slider is cheap.
  // Depends only on model+probes, so this runs once per θ, not per k change.
  $: breakdown = model && probes ? collisionBreakdown(model, probes.eval_a, probes.eval_b) : null;
  $: scores = breakdown ? breakdown.scores : null;

  $: maxK = model ? Math.max(1, Math.min(200, model.n)) : 200;
  $: k = Math.min(k, maxK);

  $: collisionSet = scores ? topK(scores, k) : [];
  // One representative draw, used for the heatmap ticks and the "ablate k
  // random instead" button (a learner ablating "random" should see one
  // concrete set of neurons actually removed, not an average of eight).
  $: randomSet = model ? randomNeurons(model.n, k, SEED) : [];
  // The full set of draws used for the *statistics* — this is what the
  // headline difference and its spread are computed from.
  $: randomSets = model ? DRAW_SEEDS.map((s) => randomNeurons(model.n, k, s)) : [];

  $: lossRows = probes
    ? [...probes.eval_a.slice(0, LOSS_ROWS_PER_SIDE), ...probes.eval_b.slice(0, LOSS_ROWS_PER_SIDE)]
    : [];

  // Teacher-forced mean cross-entropy over one sequence, mirroring the
  // next-token loss model/src/bdh_surgery/train.py:evaluate uses.
  function sequenceLoss(m, tokenIds, ablatedIdx) {
    const input = tokenIds.slice(0, -1);
    const target = tokenIds.slice(1);
    const logitsSeq = forward(m, input, { ablated: ablatedIdx });
    let total = 0;
    for (let t = 0; t < logitsSeq.length; t++) {
      const logits = logitsSeq[t];
      let maxV = -Infinity;
      for (let v = 0; v < logits.length; v++) if (logits[v] > maxV) maxV = logits[v];
      let sumExp = 0;
      for (let v = 0; v < logits.length; v++) sumExp += Math.exp(logits[v] - maxV);
      total += Math.log(sumExp) + maxV - logits[target[t]]; // -log softmax[target]
    }
    return total / logitsSeq.length;
  }

  function meanLoss(m, rows, ablatedIdx) {
    if (!m || !rows.length) return null;
    let total = 0;
    for (const row of rows) total += sequenceLoss(m, row, ablatedIdx);
    return total / rows.length;
  }

  // These were three synchronous reactives firing on every k change — six
  // forward passes over six sequences, ~1.8 s of blocked main thread each time.
  // Now one chunked async pass that yields to the browser between sequences.
  let baselineLoss = null;
  let collisionLoss = null;
  let randomLossDraws = [];
  let computing = false;
  let runId = 0;

  $: recomputeArms(model, lossRows, collisionSet, randomSets);

  async function recomputeArms(m, rows, cSet, rSets) {
    if (!m || !rows.length || !cSet.length) {
      baselineLoss = collisionLoss = null; randomLossDraws = []; return;
    }
    const mine = ++runId;
    computing = true;
    try {
      const base = await meanLossAsync(m, rows, null);
      if (mine !== runId) return;
      const coll = await meanLossAsync(m, rows, cSet);
      if (mine !== runId) return;
      const draws = [];
      for (const rs of rSets) {
        const v = await meanLossAsync(m, rows, rs);
        if (mine !== runId) return;
        draws.push(v);
      }
      baselineLoss = base; collisionLoss = coll; randomLossDraws = draws.filter((v) => v !== null);
    } finally {
      if (mine === runId) computing = false;
    }
  }

  $: randomLossMean = randomLossDraws.length
    ? randomLossDraws.reduce((a, b) => a + b, 0) / randomLossDraws.length
    : null;
  $: randomLossSD = randomLossDraws.length > 1
    ? Math.sqrt(
        randomLossDraws.reduce((a, b) => a + (b - randomLossMean) ** 2, 0) /
          (randomLossDraws.length - 1)
      )
    : null;
  // Headline: how much better targeting the collision set does than an
  // equally-sized blind guess, averaged over draws. Positive means the
  // collision set recovers more than chance; the measured result in this
  // project is that it does not — the difference is at or below zero at
  // every theta and k we checked, and even where it looks positive it sits
  // inside the random draws' own spread (randomLossSD).
  $: diff = collisionLoss !== null && randomLossMean !== null
    ? randomLossMean - collisionLoss
    : null;
  $: diffWithinNoise = diff !== null && randomLossSD !== null && Math.abs(diff) < randomLossSD;
  $: verdict = diff === null
    ? ''
    : diff <= 0
      ? 'Collision ablation does not beat random ablation here — same or worse loss.'
      : diffWithinNoise
        ? 'Collision edges out random here, but by less than the random draws’ own spread — not distinguishable from noise.'
        : 'Collision beats random here by more than the random draws’ spread.';

  $: activeArm = $ablated.length === 0
    ? 'none'
    : sameSet($ablated, collisionSet)
      ? 'collision'
      : sameSet($ablated, randomSet)
        ? 'random'
        : 'custom';

  function ablateCollision() { ablated.set(collisionSet); }
  function ablateRandom() { ablated.set(randomSet); }
  function clearAblation() { ablated.set([]); }

  function fmt(v) { return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(3); }

  // ---- 2D collision heat-map -------------------------------------------
  //
  // 1024 neurons per parent laid out row-major at 64 per row, so under concat
  // parent A occupies the top 16 rows and parent B the bottom 16 — the split is
  // structural, not decoration, and you can see at a glance whether the
  // collision set favours one parent.
  //
  // Colour jobs, kept separate: the CELL FILL is sequential (one neutral hue,
  // light to dark) because collision score is a magnitude; the RINGS are
  // categorical (the validated teal/rust pair) because collision-set and
  // random-set membership are identities. Mixing those two jobs into one
  // channel is what makes heat-maps unreadable.
  const COLS = 64;
  const COLLISION_INK = '#0a8f72';
  const RANDOM_INK = '#c25e12';

  let canvasEl;
  let wrapEl;
  let cellPx = 10;
  let hover = null;          // { idx, score, inCollision, inRandom, parent }

  $: rows = model ? Math.ceil(model.n / COLS) : 0;
  $: splitRow = model && model.n > 1024 ? 1024 / COLS : null;  // concat only

  function drawHeatmap(n, sc, collision, random, live, canvas, w) {
    if (!canvas || !n || !sc || !w) return;
    const cols = COLS;
    const rws = Math.ceil(n / cols);
    const cell = Math.max(4, Math.floor(w / cols));
    cellPx = cell;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cols * cell * dpr);
    canvas.height = Math.round(rws * cell * dpr);
    canvas.style.width = `${cols * cell}px`;
    canvas.style.height = `${rws * cell}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cols * cell, rws * cell);

    let peak = 0;
    for (let i = 0; i < n; i++) if (sc[i] > peak) peak = sc[i];
    const inv = peak > 0 ? 1 / peak : 0;

    // sequential fill: one hue, monotone light -> dark
    for (let i = 0; i < n; i++) {
      const t = sc[i] * inv;
      const x = (i % cols) * cell;
      const y = Math.floor(i / cols) * cell;
      const L = 96 - 58 * t;                 // lightness falls as score rises
      ctx.fillStyle = `hsl(168 18% ${L}%)`;
      ctx.fillRect(x, y, cell - 1, cell - 1); // 1px gap = the surface showing through
    }

    // categorical rings on top
    const ring = (idxs, colour, inset) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.5;
      for (const i of idxs) {
        if (i < 0 || i >= n) continue;
        const x = (i % cols) * cell;
        const y = Math.floor(i / cols) * cell;
        ctx.strokeRect(x + inset + 0.5, y + inset + 0.5, cell - 2 - inset * 2, cell - 2 - inset * 2);
      }
    };
    ring(random, RANDOM_INK, 0);
    ring(collision, COLLISION_INK, 0);

    // Whichever arm is running, THESE are the neurons currently zeroed. Draw
    // them knocked out — background fill plus a slash — so the heat-map shows
    // the model that actually exists, not just the two candidate sets.
    if (live && live.length) {
      const liveSet = new Set(live);
      ctx.lineWidth = 1;
      for (const i of liveSet) {
        if (i < 0 || i >= n) continue;
        const x = (i % cols) * cell;
        const y = Math.floor(i / cols) * cell;
        ctx.fillStyle = '#f7f6f3';
        ctx.fillRect(x, y, cell - 1, cell - 1);
        ctx.strokeStyle = '#9a948a';
        ctx.beginPath();
        ctx.moveTo(x + 1.5, y + 1.5);
        ctx.lineTo(x + cell - 2.5, y + cell - 2.5);
        ctx.stroke();
      }
    }

    // where parent A's neurons end and parent B's begin
    if (splitRow) {
      ctx.strokeStyle = '#1c1b19';
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, splitRow * cell - 0.5);
      ctx.lineTo(cols * cell, splitRow * cell - 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function measure() {
    if (wrapEl) drawHeatmap(model ? model.n : 0, scores, collisionSet, randomSet,
                            $ablated, canvasEl, wrapEl.clientWidth);
  }

  onMount(() => {
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  });

  $: if (canvasEl && wrapEl) drawHeatmap(model ? model.n : 0, scores, collisionSet,
                                         randomSet, $ablated, canvasEl, wrapEl.clientWidth);

  function onMove(e) {
    if (!model || !scores) return;
    const r = canvasEl.getBoundingClientRect();
    const c = Math.floor((e.clientX - r.left) / cellPx);
    const row = Math.floor((e.clientY - r.top) / cellPx);
    const idx = row * COLS + c;
    if (c < 0 || c >= COLS || idx < 0 || idx >= model.n) { hover = null; return; }
    hover = {
      idx,
      score: scores[idx],
      inCollision: collisionSet.includes(idx),
      inRandom: randomSet.includes(idx),
      isAblated: $ablated.includes(idx),
      parent: splitRow ? (idx < 1024 ? 'A' : 'B') : 'merged',
    };
  }

</script>

<div class="surgery">
  <p class="note">
    Ablation zeroes a neuron's direct output exactly, but not its whole causal footprint: the sparse
    activation also feeds attention, which sums over the entire neuron axis — so removing any neuron
    perturbs the attention pattern for every other neuron by roughly 2% of typical activation scale.
    That's intrinsic to attention, not a bug, and it's exactly why the random-ablation control below
    exists: it goes through the same perturbation, so the difference isolates the collision effect.
  </p>

  {#if !model || !probes}
    <p class="muted">waiting on the merged model…</p>
  {:else}
    <div class="heatmap-block" bind:this={wrapEl}>
      <div class="hm-frame"
           role="img"
           aria-label={`Collision score across ${model.n} merged neurons, laid out ${COLS} per row${
             splitRow ? `; parent A fills the top ${splitRow} rows and parent B the bottom ${rows - splitRow}` : ''
           }. Darker cells fire strongly on both parents' directions at once. Teal rings mark the ${k}-neuron collision set, rust rings one draw of the size-matched random control. ${
             $ablated.length ? `${$ablated.length} cells are drawn knocked out — those neurons are zeroed in the model right now.` : 'No neurons are currently zeroed.'
           }`}>
        <canvas bind:this={canvasEl}
                aria-hidden="true"
                on:mousemove={onMove}
                on:mouseleave={() => (hover = null)}></canvas>
      </div>

      <p class="hm-readout">
        {#if hover}
          neuron <strong>#{hover.idx}</strong>
          {#if splitRow}<span class="dim">· parent {hover.parent}</span>{/if}
          <span class="dim">· score {hover.score.toFixed(4)}</span>
          {#if hover.inCollision}<span class="tag collision">in collision set</span>{/if}
          {#if hover.inRandom}<span class="tag random">in random control</span>{/if}
          {#if hover.isAblated}<span class="tag off">zeroed right now</span>{/if}
        {:else}
          <span class="dim">
            {rows} rows x {COLS} neurons{splitRow ? ' — dashed line splits parent A from parent B' : ''}. Hover a cell.
          </span>
        {/if}
      </p>

      <div class="legend">
        <span class="swatch heat"></span> collision score (darker = higher)
        <span class="swatch ring-collision"></span> top-k collision set
        <span class="swatch ring-random"></span> random control set
        <span class="swatch knocked"></span>
        {activeArm === 'none' ? 'nothing zeroed' : `zeroed now (${$ablated.length})`}
      </div>
    </div>

    <div class="controls">
      <label class="k-row" for="k-slider">
        k = {k} neurons
        <input id="k-slider" type="range" min="1" max={maxK} step="1" bind:value={k} />
      </label>
      <div class="buttons">
        <button class="arm arm-collision"
                class:on={activeArm === 'collision'}
                aria-pressed={activeArm === 'collision'}
                on:click={ablateCollision}>ablate collision set</button>
        <button class="arm arm-random"
                class:on={activeArm === 'random'}
                aria-pressed={activeArm === 'random'}
                on:click={ablateRandom}>ablate {k} random instead</button>
        <button class="ghost"
                class:on={activeArm === 'none'}
                aria-pressed={activeArm === 'none'}
                on:click={clearAblation}>clear</button>
      </div>

      <p class="arm-explain">
        {#if activeArm === 'collision'}
          <strong>Running the treatment.</strong> The {k} neurons scoring highest on the collision
          metric — those firing strongly on <em>both</em> parents' directions at once — are zeroed.
          If merge damage really lives in identifiable neurons, removing exactly these should help
          more than removing any old {k}.
        {:else if activeArm === 'random'}
          <strong>Running the control.</strong> {k} neurons chosen by a seeded PRNG, ignoring the
          collision score entirely. This is the comparison that makes the treatment mean anything:
          it suffers the same loss of capacity and the same attention-pathway disturbance, just
          without targeting. Any recovery it shows is what recovery looks like <em>by luck</em>.
        {:else if activeArm === 'none'}
          <strong>No ablation.</strong> The merged model as fused, nothing removed — the baseline
          both arms are measured against.
        {:else}
          <strong>Custom set.</strong> Neither the top-{k} collision set nor the current random draw.
        {/if}
      </p>
    </div>

    {#if computing}<p class="computing-note">recomputing both arms…</p>{/if}
    <AblationStrip
      baseline={baselineLoss}
      collision={collisionLoss}
      draws={randomLossDraws}
      mean={randomLossMean}
      sd={randomLossSD} />
    <div class="deriv-wrap">
      <SurgeryDerivation
        {breakdown}
        {collisionSet}
        {k}
        baseline={baselineLoss}
        collisionLoss={collisionLoss}
        randomMean={randomLossMean}
        randomSD={randomLossSD}
        nDraws={randomLossDraws.length}
        busy={computing} />
    </div>

    <p class="sub">
      Every number above is computed live in your browser from the weights on screen. The comparison that
      matters is positional: the collision marker against the band of random draws, not the raw drop from
      baseline — a large drop means nothing if removing the same number of random neurons drops it just as far.
    </p>
  {/if}
</div>

<style>
  .arm {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--fg);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .arm:hover { border-color: var(--muted); }
  /* the active arm is filled in its own series colour, so the button and the
     mark it produced on the chart below read as the same thing */
  .arm-collision.on {
    background: #0a8f72;
    border-color: #0a8f72;
    color: #ffffff;
  }
  .arm-random.on {
    background: #c25e12;
    border-color: #c25e12;
    color: #ffffff;
  }
  .ghost.on {
    background: var(--tok-bg);
    border-color: var(--muted);
    color: var(--fg);
  }
  .arm-explain {
    margin: 0.15rem 0 0;
    font-size: 0.76rem;
    line-height: 1.5;
    color: var(--fg);
    max-width: 78ch;
    border-left: 2px solid var(--border);
    padding-left: 0.55rem;
  }
  .deriv-wrap {
    padding: 0.85rem 1rem;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .computing-note { margin: 0; font-size: 0.72rem; color: var(--muted); }
  .surgery {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
  }
  .note {
    font-size: 0.78rem;
    color: var(--muted);
    background: var(--tok-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    margin: 0;
    max-width: 72ch;
  }
  .muted {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .hm-frame { line-height: 0; }
  .hm-frame canvas {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--panel-bg);
    cursor: crosshair;
    max-width: 100%;
  }
  .hm-readout { margin: 0; font-size: 0.75rem; min-height: 1.2em; }
  .hm-readout .dim { color: var(--muted); }
  .tag {
    font-size: 0.68rem; padding: 0.03rem 0.35rem;
    border-radius: 999px; margin-left: 0.3rem;
  }
  .tag.collision { background: #e1efe9; color: #0a5f4d; }
  .tag.random { background: #f7e6d5; color: #8a4a10; }
  .tag.off { background: #e8e4dc; color: #4a4640; }
  .swatch.ring-collision {
    border: 2px solid #0a8f72; background: transparent;
  }
  .swatch.knocked {
    background: #f7f6f3;
    border: 1px solid #9a948a;
    background-image: linear-gradient(135deg, transparent 42%, #9a948a 42%, #9a948a 58%, transparent 58%);
  }
  .swatch.ring-random {
    border: 2px solid #c25e12; background: transparent;
  }
  .heatmap-block {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  canvas {
    width: 100%;
    height: 40px;
    display: block;
    border-radius: 4px;
    border: 1px solid var(--border);
    image-rendering: pixelated;
  }
  .legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    font-size: 0.72rem;
    color: var(--muted);
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-left: 0.6rem;
  }
  .swatch:first-child { margin-left: 0; }
  .swatch.heat {
    background: linear-gradient(90deg, hsl(168 18% 96%), hsl(168 18% 38%));
    border: 1px solid var(--border);
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .k-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.8rem;
    color: var(--fg);
  }
  .k-row input[type='range'] {
    flex: 1;
    accent-color: var(--accent);
  }
  .buttons {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  button {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--fg);
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
    cursor: pointer;
  }
  button.ghost {
    background: transparent;
    color: var(--muted);
  }

  .sub {
    color: var(--muted);
    font-size: 0.78rem;
    margin: 0;
    max-width: 68ch;
  }

  @media (max-width: 760px) {
  }
</style>
