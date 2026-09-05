<script>
  import { onMount } from 'svelte';
  import { loadModel, forward } from './bdh_forward.js';
  import { manifest, dataset, theta, mergeMode, ablated, parentA, parentB, mergedModel, narrativeStep } from './lib/store.js';
  import OutputPanels from './lib/OutputPanels.svelte';
  import PhaseDiagram from './lib/PhaseDiagram.svelte';
  import Surgery from './lib/Surgery.svelte';
  import Narrative from './lib/Narrative.svelte';
  import Derivation from './lib/Derivation.svelte';
  import FusionStrip from './lib/FusionStrip.svelte';
  import HonestyBadge from './lib/HonestyBadge.svelte';
  import { buildLexicon, labelSequence, splitProbe, layoutFor } from './lib/tokens.js';
  import { greedyDecode } from './lib/generate.js';
  import { measureDamageAsync, cachedDamage, damageKey } from './lib/compute.js';

  let probes = null;
  let sweep = null;
  let loadErr = null;
  let outputs = null;
  let sweepTheta = 0.5;
  let precomputedPoint = null;
  let phaseXMode = 'theta';
  let reqToken = 0;

  // `kind` matters: weights ship as int8 and loadModel dequantizes them, so the
  // bytes must be viewed as Int8Array. Viewing them as Float32Array yields a
  // quarter-length array and the merge fails with an out-of-bounds offset.
  // Vite injects the deploy base here, so the same build works at a domain
  // root and under a subpath like GitHub Pages' /<repo>/.
  const DATA = `${import.meta.env.BASE_URL}data`.replace(/\/{2,}/g, '/');

  const fetchBin = (file, kind) =>
    fetch(`${DATA}/${file}`)
      .then((r) => r.arrayBuffer())
      .then((buf) => (kind === 'int8' ? new Int8Array(buf) : new Float32Array(buf)));

  let probesAllDatasets = null;
  let sweepAllDatasets = null;

  onMount(async () => {
    try {
      const [m, p, s] = await Promise.all([
        fetch(`${DATA}/manifest.json`).then((r) => r.json()),
        fetch(`${DATA}/probes.json`).then((r) => r.json()),
        fetch(`${DATA}/sweep.json`).then((r) => r.json()),
      ]);
      manifest.set(m);
      probesAllDatasets = p.datasets;
      sweepAllDatasets = s.datasets;
      const datasetIds = Object.keys(m.datasets);
      dataset.set(datasetIds.includes('baseline') ? 'baseline' : datasetIds[0]);

      const featuredKeys = Object.keys(m.datasets[$dataset].featured);
      const defaultKey = featuredKeys.includes('0.5') ? '0.5' : featuredKeys[0];
      theta.set(defaultKey);
      sweepTheta = parseFloat(defaultKey);
    } catch (err) {
      loadErr = String(err);
    }
  });

  // The manifest fragment for the dataset on screen — same {config, featured}
  // shape loadModel already expects, so passing THIS instead of the whole
  // manifest is the entire change needed to make loadModel dataset-aware.
  $: curDataset = $manifest && $manifest.datasets[$dataset] ? $manifest.datasets[$dataset] : null;
  $: curSweep = sweepAllDatasets ? sweepAllDatasets[$dataset] : null;
  $: probes = probesAllDatasets ? probesAllDatasets[$dataset] : null;
  // kept for the parts of the template that still say `sweep` — one dataset's
  // slice, never the whole file.
  $: sweep = curSweep;

  // Datasets differ in n_concepts (and sometimes vocab_size), so switching one
  // is a full reload — new parents, a fresh collision analysis, everything
  // downstream recomputes. θ itself is left alone: all three datasets export
  // the same 11 θ keys, so staying on "0.5" across a dataset switch is what
  // lets a learner compare the same θ under two different vocabularies.
  const datasetOptions = () =>
    $manifest ? Object.entries($manifest.datasets).map(([id, d]) => ({ id, label: d.label, blurb: d.blurb })) : [];

  function selectDataset(id) {
    if (id === $dataset) return;
    ablated.set([]);
    outputs = null;
    sharedDamage = null;
    dataset.set(id);
  }

  // All 11 θ ship weights now, so a slider drag can ask for many of them in a
  // second. Cache by (dataset, θ, side) so each is fetched and dequantized
  // exactly once — and so the SAME θ string in two different datasets is
  // never confused for the same model.
  const modelCache = new Map();

  async function loadSide(m, ds, th, side) {
    const key = `${ds}|${th}|${side}`;
    if (!modelCache.has(key)) modelCache.set(key, loadModel(m, th, side, fetchBin));
    return modelCache.get(key);
  }

  async function loadParents(m, ds, th) {
    const myToken = ++reqToken;
    const [a, b] = await Promise.all([loadSide(m, ds, th, 'A'), loadSide(m, ds, th, 'B')]);
    if (myToken !== reqToken) return; // superseded by a later θ/dataset change
    parentA.set(a);
    parentB.set(b);
  }

  $: if (curDataset && $dataset && $theta) loadParents(curDataset, $dataset, $theta);

  function regenerate() {
    const featured = curDataset.featured[$theta];
    const lexicon = buildLexicon(featured);
    // large_vocab's specials (BOS/SEP/EOS) live at different ids than the
    // 24-concept datasets' — everything below has to use THIS dataset's
    // layout, or splitProbe silently finds no SEP/EOS and greedyDecode never
    // stops early.
    const layout = layoutFor(nConcepts);
    const probeSeq = curProbes.eval_a[0];
    const { prompt, target } = splitProbe(probeSeq, layout);
    const maxNew = target.length || 4;

    const contA = greedyDecode($parentA, prompt, maxNew, { eos: layout.eos });
    const contB = greedyDecode($parentB, prompt, maxNew, { eos: layout.eos });
    const mergedAblated = $ablated.length ? $ablated : null;
    const contM = greedyDecode($mergedModel, prompt, maxNew, { ablated: mergedAblated, eos: layout.eos });

    outputs = {
      a: { prompt: labelSequence(prompt, lexicon, layout), continuation: labelSequence(contA, lexicon, layout) },
      b: { prompt: labelSequence(prompt, lexicon, layout), continuation: labelSequence(contB, lexicon, layout) },
      merged: { prompt: labelSequence(prompt, lexicon, layout), continuation: labelSequence(contM, lexicon, layout) },
      oracle: { prompt: labelSequence(prompt, lexicon, layout), target: labelSequence(target, lexicon, layout) },
    };
  }

  // $ablated is named here (not just read inside regenerate()) because Svelte
  // only re-runs a labeled reactive statement when a store is referenced
  // *syntactically* in that statement — omitting it means an ablation change
  // silently fails to refresh the output panels.
  $: if ($parentA && $parentB && $mergedModel && curProbes && curDataset && $ablated) regenerate();

  function matchFeaturedKey(ds, val) {
    if (!ds) return null;
    for (const k of Object.keys(ds.featured)) {
      if (Math.abs(parseFloat(k) - val) < 1e-6) return k;
    }
    return null;
  }

  function findSweepPoint(sw, val) {
    if (!sw) return null;
    const rows = sw.points.filter((p) => Math.abs(p.theta - val) < 1e-6);
    if (!rows.length) return null;
    const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
    return {
      theta: val,
      m_mean: mean(rows.map((r) => r.m_mean)),
      d_mean: mean(rows.map((r) => r.d_mean)),
      d_avg_arm: mean(rows.map((r) => r.d_avg_arm)),
      seeds: rows.length,
    };
  }

  // The active dataset's own concept count — 24 for baseline and long_phrase,
  // 40 for large_vocab. Never hard-coded, so the Venn/label text stays correct
  // when the learner switches datasets.
  $: nConcepts = curDataset ? curDataset.n_concepts : 24;

  // Shared concept count for the θ on screen — read from the real lexicons
  // when that θ ships them, else the same round(θ·n_concepts) rule
  // build_lexicons uses.
  $: sharedConcepts = (() => {
    if (!curDataset) return null;
    const key = matchFeaturedKey(curDataset, sweepTheta);
    if (key) {
      const f = curDataset.featured[key];
      return f.lex_a.filter((t, i) => t === f.lex_b[i]).length;
    }
    return Math.round(sweepTheta * nConcepts);
  })();

  // Single entry point for changing θ, so Act 1's slider and Act 3's plot stay
  // in step and both actually reload the weights for the θ they claim to show.
  // sweepTheta updates instantly so the slider and its readout stay responsive.
  // Committing to the `theta` store is what reloads weights and restarts every
  // heavy view, so it is debounced — otherwise one drag across the slider fires
  // eleven model loads and eleven timeline restarts.
  let commitTimer = null;
  const COMMIT_MS = 140;

  function selectTheta(val, immediate = false) {
    sweepTheta = val;
    if (!curDataset) return;
    const key = matchFeaturedKey(curDataset, val);
    if (!key) return;
    clearTimeout(commitTimer);
    if (immediate) { theta.set(key); return; }
    commitTimer = setTimeout(() => theta.set(key), COMMIT_MS);
  }

  function onPhaseSelect(e) {
    selectTheta(e.detail);
  }

  // Splits at parent A's own n (pre-merge), since concat glues A's neurons
  // then B's; under average the merged n equals a parent's n and there is
  // no split.
  $: splitAt = $mergeMode === 'concat' && $parentA ? $parentA.n : null;
  // probes.json is keyed by (dataset, theta): each theta has its own target
  // lexicons, so scoring a theta=0.5 parent against theta=0.0's corpus would
  // measure nothing. All 11 θ ship weights for every dataset now, so
  // hasWeightsForSelected is nearly always true — it stays as a guard for
  // whichever θ a future dataset might not cover.
  $: if (curSweep) precomputedPoint = findSweepPoint(curSweep, sweepTheta);
  $: selectedKey = curDataset ? matchFeaturedKey(curDataset, sweepTheta) : null;
  $: hasWeightsForSelected = selectedKey !== null;
  $: curProbes = probes && $theta && probes.featured && probes.featured[$theta]
    ? { pivot: probes.pivot, ...probes.featured[$theta] }
    : null;
  $: seqLen = curProbes ? curProbes.eval_a[0].length : null;

  // ---- The phase diagram's one live mark ---------------------------------
  //
  // The 33 plotted points came from training 66 models offline and cannot be
  // recomputed here. But damage itself is just merged-loss minus parent-loss,
  // and every model in that subtraction IS loaded — so we can measure it in
  // the browser for the θ currently on screen, under whatever merge mode and
  // ablation set the learner has chosen, and drop it onto the same axes.
  // Same definition as model/src/bdh_surgery/damage.py: mean over the two
  // directions, each parent scored on its own direction.
  // ONE damage computation per (θ, merge rule), shared by the derivation panel
  // and the phase diagram's live marker. They were each running their own —
  // the same forward passes twice, ~1.7 s of duplicated blocking work.
  const LIVE_ROWS = 8;
  let sharedDamage = null;
  let damageBusy = false;

  $: damageInputs = {
    ds: $dataset,
    th: $theta,
    mode: $mergeMode,
    a: $parentA,
    b: $parentB,
    m: $mergedModel,
    p: curProbes,
  };

  $: recomputeDamage(damageInputs);

  async function recomputeDamage(inp) {
    if (!inp.a || !inp.b || !inp.m || !inp.p || !inp.th || !inp.ds) { sharedDamage = null; return; }
    const key = damageKey(inp.ds, inp.th, inp.mode, null);
    damageBusy = true;
    try {
      const res = await cachedDamage(key, () =>
        measureDamageAsync(inp.m, inp.a, inp.b,
          inp.p.eval_a.slice(0, LIVE_ROWS), inp.p.eval_b.slice(0, LIVE_ROWS), null));
      // a newer θ/dataset may have landed while we were yielding
      if (damageKey($dataset, $theta, $mergeMode, null) === key) sharedDamage = res;
    } finally {
      damageBusy = false;
    }
  }

  $: livePoint = (() => {
    if (!sharedDamage || !curSweep) return null;
    const th = $theta ? parseFloat($theta) : sweepTheta;
    const parts = [$mergeMode];
    if ($ablated.length) parts.push(`−${$ablated.length}n`);
    // Pass θ, not a screen x: PhaseDiagram owns its own axis mode and derives
    // the position itself, so the two can't disagree.
    return { theta: th, y: sharedDamage.dMean, label: `your run · ${parts.join(' · ')}` };
  })();

  // Guided-narrative gating: Act `n` (and everything below it) is locked
  // until the learner has stepped past it, or has reached free exploration.
  // Act 1 is never locked — the page opens mid-Act-1 already generating.
  $: isLocked = (n) => $narrativeStep !== 'free' && $narrativeStep < n;
</script>

<div class="app">
  <header>
    <h1>Model Surgery: fusing two BDH language models</h1>
    <p class="tagline">
      Two small language models are trained on related but distinct synthetic languages, sharing a slice of
      vocabulary controlled by θ (0 = disjoint, 1 = identical). Fuse them and watch what survives.
    </p>
    {#if curDataset}
      <p class="caps">
        n per parent: <strong>{curDataset.config.n}</strong>
        &nbsp;·&nbsp; merged n: <strong>{$mergedModel ? $mergedModel.n : '—'}</strong>
        &nbsp;·&nbsp; sequence length: <strong>{seqLen ?? '—'}</strong>
        &nbsp;·&nbsp; vocab size: <strong>{curDataset.config.vocab_size}</strong>
      </p>
    {/if}
    {#if $manifest}
      <div class="dataset-picker" role="group" aria-label="Dataset">
        {#each datasetOptions() as opt}
          <button
            class:active={opt.id === $dataset}
            aria-pressed={opt.id === $dataset}
            title={opt.blurb}
            on:click={() => selectDataset(opt.id)}>{opt.label}</button>
        {/each}
      </div>
      {#if curDataset}<p class="dataset-blurb">{curDataset.blurb}</p>{/if}
    {/if}
  </header>

  {#if loadErr}
    <p class="error">Failed to load data: {loadErr}</p>
  {/if}

  <Narrative />

  <section>
    <div class="section-head">
      <h2>Act 1 — Fuse</h2>
      <span class="badges">
        <HonestyBadge kind="precomputed" label="parent weights" />
        <HonestyBadge kind="live" label="merge" />
      </span>
    </div>
    <p class="sub">
      Watch the fusion happen. Drag θ to change how much vocabulary the two parents share
      before they are fused — at θ = 0 they name every concept differently, at θ = 1 they
      speak the same language.
    </p>

    {#if $parentA && $parentB}
      <label class="theta-row" for="act1-theta">
        <span class="theta-cap">
          θ = {sweepTheta.toFixed(1)}
          <span class="theta-sub">
            {sharedConcepts === null ? '' : `${sharedConcepts} of ${nConcepts} concepts shared`}
          </span>
        </span>
        <input id="act1-theta" type="range" min="0" max="10" step="1"
               value={Math.round(sweepTheta * 10)}
               on:input={(e) => selectTheta(Number(e.target.value) / 10)} />
      </label>
      <FusionStrip parentA={$parentA} parentB={$parentB}
        mergeMode={$mergeMode} theta={$theta}
        shared={sharedConcepts} totalConcepts={nConcepts}
        damage={sharedDamage} />
    {:else}
      <p class="muted">loading parents…</p>
    {/if}


  </section>

  <section class="gated" class:locked={isLocked(2)} inert={isLocked(2)}>
    <div class="section-head">
      <h2>Act 2 — Concat vs. average <HonestyBadge kind="live" /></h2>
      <div class="segmented" role="group" aria-label="Merge mode">
        <button
          class:active={$mergeMode === 'concat'}
          aria-pressed={$mergeMode === 'concat'}
          on:click={() => mergeMode.set('concat')}>Concat</button>
        <button
          class:active={$mergeMode === 'average'}
          aria-pressed={$mergeMode === 'average'}
          on:click={() => mergeMode.set('average')}>Average</button>
      </div>
    </div>
    <p class="sub">Same prompt, four models: parent A, parent B, the merge, and the oracle target.</p>
    {#if outputs}
      <OutputPanels parentA={outputs.a} parentB={outputs.b} merged={outputs.merged} oracle={outputs.oracle} />
    {:else}
      <p class="muted">generating…</p>
    {/if}
    {#if isLocked(2)}<p class="lock-note">reach Act 2 in the narrative above to unlock this control</p>{/if}
  </section>

  <section class="gated" class:locked={isLocked(3)} inert={isLocked(3)}>
    <h2>Act 3 — The phase diagram <HonestyBadge kind="precomputed" /></h2>
    <p class="sub">
      Damage (d_mean) across the θ sweep, with the per-seed min–max spread as an error bar. Switch the x axis
      between θ (designed overlap — where the strong relationship is) and m_mean (measured overlap — where it
      isn't). The muted series is the averaging control; it sits <em>above</em> concat at every θ, because
      averaging always hurts more.
    </p>
    {#if sweep}
      <PhaseDiagram points={sweep.points} yKey="d_mean" marker={sweepTheta}
        bind:xMode={phaseXMode} {livePoint} on:select={onPhaseSelect} />

      {#if curDataset}
        <div class="derivation-wrap">
          <Derivation
            theta={$theta ? parseFloat($theta) : sweepTheta}
            lexA={selectedKey ? curDataset.featured[selectedKey].lex_a : null}
            lexB={selectedKey ? curDataset.featured[selectedKey].lex_b : null}
            nConcepts={nConcepts}
            parentA={$parentA}
            parentB={$parentB}
            merged={$mergedModel}
            mergeMode={$mergeMode}
            hasWeights={hasWeightsForSelected}
            precomputed={precomputedPoint}
            damage={sharedDamage}
            busy={damageBusy} />
        </div>
      {/if}
      {#if precomputedPoint}
        <p class="point-readout">
          θ = {precomputedPoint.theta.toFixed(2)}
          &nbsp;·&nbsp; m_mean {precomputedPoint.m_mean.toFixed(3)}
          &nbsp;·&nbsp; d_mean {precomputedPoint.d_mean.toFixed(3)}
          &nbsp;·&nbsp; d_avg_arm {precomputedPoint.d_avg_arm.toFixed(3)}
          &nbsp;·&nbsp; {precomputedPoint.seeds} seed{precomputedPoint.seeds === 1 ? '' : 's'}
          <!-- Always precomputed: these three come straight out of sweep.json and are
               never recomputed in the browser, whatever θ is selected. -->
          <HonestyBadge kind="precomputed" label="precomputed (sweep.json)" />
        </p>
      {/if}
    {:else}
      <p class="muted">loading sweep…</p>
    {/if}
    {#if isLocked(3)}<p class="lock-note">reach Act 3 in the narrative above to unlock this control</p>{/if}
  </section>

  <section class="gated" class:locked={isLocked(4)} inert={isLocked(4)}>
    <h2>Act 4 — Surgery <HonestyBadge kind="live" /></h2>
    <p class="sub">
      Find the neurons where the merge collides — active on both parent directions at once — ablate
      them, and watch the loss move. A size-matched random ablation runs alongside so the collision
      story has to earn its keep against chance. It doesn't: the difference below is negative at
      most settings, which is the honest answer to the second half of our original hypothesis.
    </p>
    {#if $mergedModel && probes}
      <Surgery model={$mergedModel} probes={curProbes} />
    {:else}
      <p class="muted">waiting on the merged model…</p>
    {/if}
    {#if isLocked(4)}<p class="lock-note">reach Act 4 in the narrative above to unlock this control</p>{/if}
  </section>

  <footer class="caps-footer">
    <strong>Toy-scale reimplementation — not an official BDH model.</strong>
    n = {curDataset ? curDataset.config.n : 1024} per parent, {$mergedModel ? $mergedModel.n : 2048} merged,
    sequence length {seqLen ?? 16}, dataset "{curDataset ? curDataset.label : '—'}". Two synthetic languages
    over a shared pivot, not natural text — see README.md for the full honesty table and known limitations.
  </footer>
</div>

<style>
  :global(:root) {
    --bg: #f7f6f3;
    --fg: #1c1b19;
    --muted: #726f68;
    --border: #ddd9d1;
    --panel-bg: #ffffff;
    --tok-bg: #efece5;
    --accent: #1f6f5c;
    --accent-weak: #e1efe9;
  }
  :global(html, body) {
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  }
  .app {
    max-width: 920px;
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
    display: flex;
    flex-direction: column;
    gap: 2.25rem;
  }
  header h1 {
    font-size: 1.4rem;
    margin: 0 0 0.4rem;
  }
  .tagline {
    color: var(--muted);
    font-size: 0.9rem;
    max-width: 62ch;
    margin: 0 0 0.6rem;
  }
  .caps {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0;
  }
  .dataset-picker {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
  }
  .dataset-picker button {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--fg);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    cursor: pointer;
  }
  .dataset-picker button.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }
  .dataset-blurb {
    font-size: 0.76rem;
    color: var(--muted);
    margin: 0.3rem 0 0;
    max-width: 62ch;
  }
  section h2 {
    font-size: 1.05rem;
    margin: 0 0 0.25rem;
  }
  .sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0 0 0.75rem;
    max-width: 68ch;
  }
  .section-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .muted {
    color: var(--muted);
    font-size: 0.85rem;
  }
  .error {
    color: #a5453b;
  }
  .segmented {
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
  }
  .segmented button {
    border: none;
    background: transparent;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    cursor: pointer;
    color: var(--muted);
  }
  .segmented button.active {
    background: var(--accent);
    color: #ffffff;
  }
  .theta-row {
    display: flex; align-items: center; gap: 0.85rem;
    font-size: 0.8rem; margin: 0.1rem 0 0.55rem;
  }
  .theta-row input { flex: 1; max-width: 320px; }
  .theta-cap { min-width: 15rem; }
  .theta-sub { color: var(--muted); font-size: 0.74rem; margin-left: 0.4rem; }
  .derivation-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    margin-top: 0.9rem;
    padding: 0.85rem 1rem;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .point-readout {
    font-size: 0.78rem;
    color: var(--fg);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.1rem;
    margin-top: 0.6rem;
  }
  .point-readout :global(.honesty-badge) {
    margin-left: 0.5rem;
  }
  section h2 :global(.honesty-badge) {
    margin-left: 0.5rem;
    vertical-align: middle;
  }
  .badges {
    display: flex;
    gap: 0.35rem;
  }
  .gated {
    position: relative;
    transition: opacity 0.2s ease;
  }
  .gated.locked {
    opacity: 0.45;
    pointer-events: none;
    user-select: none;
  }
  .lock-note {
    margin: 0.6rem 0 0;
    font-size: 0.72rem;
    color: var(--muted);
    font-style: italic;
  }
  .caps-footer {
    border-top: 1px solid var(--border);
    padding-top: 1rem;
    font-size: 0.72rem;
    color: var(--muted);
    max-width: 74ch;
    line-height: 1.5;
  }
  .caps-footer strong {
    color: var(--fg);
  }
</style>
