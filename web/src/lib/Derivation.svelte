<script>
  // Walks the learner from "θ = 0.5" to the dot on the phase diagram, one real
  // quantity at a time. Nothing here is scripted: the Venn counts come from the
  // manifest's actual per-θ lexicons, and every loss is a live forward pass on
  // the weights currently loaded. The animation only controls *when* each real
  // number is revealed, never what it is.
  import { createEventDispatcher, onDestroy, tick } from 'svelte';
  import gsap from 'gsap';


  export let theta = null;      // numeric θ on screen
  export let lexA = null;       // manifest.featured[θ].lex_a
  export let lexB = null;       // manifest.featured[θ].lex_b
  export let parentA = null;
  export let parentB = null;
  export let merged = null;
  export let mergeMode = 'concat';
  // Only 3 of the 11 swept θ ship weights (they are ~800 kB each). For the
  // other 8 the lexicon maths is still exact — it needs only θ — but the losses
  // cannot be recomputed here, so we show the sweep's own figure and say so.
  export let hasWeights = true;
  export let precomputed = null;   // { d_mean, seeds } for this θ, or null
  export let nConcepts = 24;
  // Computed once in App and shared with the phase diagram's live marker —
  // this panel used to run the same forward passes a second time.
  export let damage = null;
  export let busy = false;

  const dispatch = createEventDispatcher();
  const ROWS = 8;               // all the probe rows we ship; closer to the offline figure
  const STEP_MS = 900;

  const A_COLOR = '#0a8f72';
  const B_COLOR = '#c25e12';

  let step = 0;                 // 0 idle · 1 venn · 2 merge · 3 dA · 4 dB · 5 combine
  const LAST = 5;
  let tl = null;

  // Numbers are tweened toward their real values so the formula reads as
  // *evaluating*. The tween only controls the intermediate frames — every
  // timeline lands with an explicit set() to the exact computed figure, so
  // what you finally read is never a rounding artifact of the animation.
  let anim = { frac: 0, dA: 0, dB: 0, dMean: 0, n: 0 };

  const reduceMotion =
    typeof window !== 'undefined' &&
    window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- exact, from the real lexicons -------------------------------------
  // build_lexicons() assigns a shared surface token to a θ fraction of the 24
  // concepts, so "same token at the same concept index" IS the intersection.
  // Counted from the real lexicons when we have them; otherwise derived from θ
  // by the same rule build_lexicons() uses, which is exact either way.
  $: total = lexA ? lexA.length : nConcepts;
  $: shared = lexA && lexB
    ? lexA.filter((t, i) => t === lexB[i]).length
    : theta === null ? null : Math.round(theta * nConcepts);
  $: onlyA = shared !== null ? total - shared : null;
  $: frac = shared !== null && total ? shared / total : 0;

  // ---- exact, from live forward passes ------------------------------------
  $: dmg = hasWeights ? damage : null;
  // What the last step reports: the live figure when we can compute it, the
  // sweep's precomputed mean when we cannot.
  $: finalD = dmg ? dmg.dMean : precomputed ? precomputed.d_mean : null;

  // Re-arm whenever the inputs change, so selecting a θ replays the derivation.
  $: if (theta !== null && !busy) restart(theta, mergeMode, hasWeights, dmg, precomputed);

  async function restart() {
    if (tl) tl.kill();
    step = 0;
    anim = { frac: 0, dA: 0, dB: 0, dMean: 0, n: 0 };
    await tick();

    if (!dmg) {
      // No weights for this θ: animate the exact part (the Venn), then show the
      // precomputed damage rather than inventing a live number.
      if (shared === null) return;
      if (reduceMotion) { step = LAST; anim = { ...anim, frac, dMean: finalD ?? 0 }; return; }
      tl = gsap.timeline();
      tl.to(anim, {
        frac, duration: (STEP_MS / 1000) * 0.9, ease: 'power2.inOut',
        onStart: () => (step = 1), onUpdate: () => (anim = anim),
      });
      tl.to(anim, {
        dMean: finalD ?? 0, duration: (STEP_MS / 1000) * 0.9, ease: 'power2.out',
        onStart: () => (step = LAST), onUpdate: () => (anim = anim),
        onComplete: () => { anim = { ...anim, frac, dMean: finalD ?? 0 }; },
      }, `+=${(STEP_MS / 1000) * 0.25}`);
      return;
    }

    const target = {
      frac,
      dA: dmg.dA,
      dB: dmg.dB,
      dMean: dmg.dMean,
      n: merged ? merged.n : 0,
    };

    if (reduceMotion) {
      // Honour the OS preference: no motion, same real numbers, same order.
      step = LAST;
      anim = { ...target };
      dispatch('done', { theta, d: dmg.dMean });
      return;
    }

    const D = STEP_MS / 1000;
    tl = gsap.timeline({
      onComplete: () => {
        anim = { ...target };            // land exactly, never on a tween frame
        dispatch('done', { theta, d: dmg.dMean });
      },
    });

    // 1 — the circles slide to their real overlap while the count runs up
    tl.set({}, {}, 0.15)
      .to(anim, {
        frac: target.frac,
        duration: D * 0.9,
        ease: 'power2.inOut',
        onStart: () => (step = 1),
        onUpdate: () => (anim = anim),
      });

    // 2 — the merged neuron axis
    tl.to(anim, {
      n: target.n,
      duration: D * 0.6,
      ease: 'power1.out',
      onStart: () => (step = 2),
      onUpdate: () => (anim = anim),
    }, `+=${D * 0.25}`);

    // 3, 4 — each direction's damage counts out from zero
    tl.to(anim, {
      dA: target.dA,
      duration: D * 0.8,
      ease: 'power1.out',
      onStart: () => (step = 3),
      onUpdate: () => (anim = anim),
    }, `+=${D * 0.25}`);

    tl.to(anim, {
      dB: target.dB,
      duration: D * 0.8,
      ease: 'power1.out',
      onStart: () => (step = 4),
      onUpdate: () => (anim = anim),
    }, `+=${D * 0.2}`);

    // 5 — the average, and the point lands
    tl.to(anim, {
      dMean: target.dMean,
      duration: D * 0.9,
      ease: 'power2.out',
      onStart: () => (step = 5),
      onUpdate: () => (anim = anim),
    }, `+=${D * 0.2}`);
  }

  function replay() {
    restart();
  }

  onDestroy(() => { if (tl) tl.kill(); });

  const f = (v, p = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(p));

  // Venn geometry: circle separation is driven by the real overlap fraction, so
  // the picture and the count can never disagree.
  const R = 46;
  $: sep = 2 * R * (1 - anim.frac);     // 2R apart at θ=0, coincident at θ=1
  $: cxA = 160 - sep / 2;
  $: cxB = 160 + sep / 2;
</script>

<div class="deriv">
  <div class="deriv-head">
    <span class="title">
      How θ = {theta === null ? '—' : theta.toFixed(1)} becomes a point on the plot
      <span class="mode {busy ? 'pre' : hasWeights ? 'live' : 'pre'}">
        {busy ? 'computing…' : hasWeights ? 'live' : 'weights not shipped for this θ'}
      </span>
    </span>
    <button class="replay" on:click={replay} aria-label="Replay the derivation">↻ replay</button>
  </div>

  <ol class="steps">
    <!-- 1 · the Venn, straight from the lexicons -->
    <li class:on={step >= 1}>
      <span class="n">1</span>
      <div class="body">
        <p class="lead">θ sets how many of the {total ?? '—'} concepts share a surface token.</p>
        <div class="venn-row">
          <svg viewBox="0 0 320 118" role="img"
               aria-label={`Venn diagram: language A and language B each name ${total} concepts, sharing ${shared} of them at theta ${theta}.`}>
            <circle cx={cxA} cy="59" r={R} fill={A_COLOR} opacity="0.30" />
            <circle cx={cxB} cy="59" r={R} fill={B_COLOR} opacity="0.30" />
            <circle cx={cxA} cy="59" r={R} fill="none" stroke={A_COLOR} stroke-width="2" />
            <circle cx={cxB} cy="59" r={R} fill="none" stroke={B_COLOR} stroke-width="2" />
            <text x={cxA - R / 2} y="64" class="vlbl" text-anchor="middle" fill={A_COLOR}>{onlyA ?? '—'}</text>
            <text x={cxB + R / 2} y="64" class="vlbl" text-anchor="middle" fill={B_COLOR}>{onlyA ?? '—'}</text>
            {#if shared}
              <text x="160" y="64" class="vlbl shared" text-anchor="middle">{Math.round(anim.frac * total)}</text>
            {/if}
            <text x={cxA - R / 2} y="22" class="vcap" text-anchor="middle" fill={A_COLOR}>A only</text>
            <text x={cxB + R / 2} y="22" class="vcap" text-anchor="middle" fill={B_COLOR}>B only</text>
          </svg>
          <p class="calc">
            <code>|A ∩ B| = round(θ × {total ?? '—'}) = <strong>{shared ?? '—'}</strong></code>
            <span class="hint">counted from the manifest's own lexicons, not from θ</span>
          </p>
        </div>
      </div>
    </li>

    {#if hasWeights}
    <!-- 2 · the merge -->
    <li class:on={step >= 2}>
      <span class="n">2</span>
      <div class="body">
        <p class="lead">
          Both parents are fused by <strong>{mergeMode}</strong>.
        </p>
        <p class="calc">
          <code>
            n = {parentA ? parentA.n : '—'} {mergeMode === 'concat' ? '+' : '⊕'} {parentB ? parentB.n : '—'}
            → <strong>{merged ? Math.round(anim.n) : '—'}</strong>
          </code>
          <span class="hint">
            {mergeMode === 'concat'
              ? 'concatenation grows the neuron axis; both parents stay intact'
              : 'averaging keeps the axis the same size and blends the two'}
          </span>
        </p>
      </div>
    </li>

    <!-- 3 · direction A -->
    <li class:on={step >= 3}>
      <span class="n">3</span>
      <div class="body">
        <p class="lead">Score the merged model on A's own direction, against A's parent.</p>
        <p class="calc">
          <code>
            D<sub>A</sub> = {f(dmg?.mergedA)} − {f(dmg?.parentA)} =
            <strong style="color:{A_COLOR}">{dmg ? (anim.dA >= 0 ? '+' : '') + f(anim.dA) : '—'}</strong>
          </code>
          <span class="hint">live forward passes over {ROWS} held-out sequences</span>
        </p>
      </div>
    </li>

    <!-- 4 · direction B -->
    <li class:on={step >= 4}>
      <span class="n">4</span>
      <div class="body">
        <p class="lead">Same for B. Each parent is only ever scored on the direction it was trained for.</p>
        <p class="calc">
          <code>
            D<sub>B</sub> = {f(dmg?.mergedB)} − {f(dmg?.parentB)} =
            <strong style="color:{B_COLOR}">{dmg ? (anim.dB >= 0 ? '+' : '') + f(anim.dB) : '—'}</strong>
          </code>
          <span class="hint">the two rarely match — that gap is the directional asymmetry</span>
        </p>
      </div>
    </li>

    {:else}
    <li class:on={step >= 1} class="skipped">
      <span class="n">–</span>
      <div class="body">
        <p class="lead">
          Steps 2–4 need this θ's trained weights, and only θ = 0.0, 0.5 and 1.0 ship them
          (~800 kB per parent). The lexicon maths above is exact for every θ; the damage below
          is the sweep's own figure, averaged over its {precomputed ? precomputed.seeds : '—'} seeds.
        </p>
      </div>
    </li>
    {/if}

    <!-- 5 · the point -->
    <li class:on={step >= 5} class="final">
      <span class="n">5</span>
      <div class="body">
        <p class="lead">
          {hasWeights
            ? 'Average the two directions. That is the y of the point on the plot.'
            : 'The sweep already measured this θ. That value is the y of the point on the plot.'}
        </p>
        <p class="calc big">
          {#if hasWeights}
          <code>
            D = ({dmg ? (anim.dA >= 0 ? '+' : '') + f(anim.dA) : '—'} {dmg && anim.dB >= 0 ? '+' : '−'}
            {dmg ? f(Math.abs(anim.dB)) : '—'}) / 2 =
            <strong>{dmg ? (anim.dMean >= 0 ? '+' : '') + f(anim.dMean) : '—'}</strong>
          </code>
          {:else}
          <code>
            D = <strong>{finalD === null ? '—' : (anim.dMean >= 0 ? '+' : '') + f(anim.dMean)}</strong>
          </code>
          <span class="hint">precomputed · mean of {precomputed ? precomputed.seeds : '—'} seeds</span>
          {/if}
        </p>
        <p class="landed">
          → plotted at (θ = {theta === null ? '—' : theta.toFixed(1)},
          D = {finalD === null ? '—' : f(anim.dMean)})
        </p>
        {#if hasWeights}
          <p class="hint reconcile">
            This is the <strong>orange live marker</strong> on the plot, not a point on the curve — and it
            should not sit exactly on it. The curve averages 3 seeds over 256 held-out sequences each;
            this recomputes seed 0 alone over the 8 sequences shipped with the artifact. Same
            definition, smaller sample, one seed.
          </p>
        {/if}
      </div>
    </li>
  </ol>

  <p class="foot">
    {#if hasWeights}
      Every figure above is computed in your browser from the weights on screen — the animation
      controls only the order things appear in, never the values.
    {:else}
      The Venn counts above are exact for this θ; the damage is read from the sweep, not recomputed.
    {/if}
    The plotted curve behind it comes from 66 models trained offline, which is why it cannot be
    recomputed here.
  </p>
</div>

<style>
  .deriv { display: flex; flex-direction: column; gap: 0.6rem; }
  .deriv-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
  .title { font-size: 0.85rem; font-weight: 600; }
  .replay {
    border: 1px solid var(--border); background: var(--panel-bg); color: var(--muted);
    border-radius: 5px; padding: 0.15rem 0.5rem; font-size: 0.72rem; cursor: pointer;
  }
  .replay:hover { color: var(--fg); }
  .steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.5rem; }
  .steps li {
    display: flex; gap: 0.6rem; align-items: flex-start;
    opacity: 0.25; transform: translateY(4px);
    transition: opacity 320ms ease, transform 320ms ease;
    border-left: 2px solid transparent; padding-left: 0.5rem;
  }
  .steps li.on { opacity: 1; transform: none; border-left-color: var(--border); }
  .steps li.final.on { border-left-color: var(--accent); }
  .n {
    flex: 0 0 auto; width: 18px; height: 18px; border-radius: 50%;
    background: var(--tok-bg); color: var(--muted);
    font-size: 0.66rem; display: grid; place-items: center; margin-top: 0.12rem;
  }
  .steps li.on .n { background: var(--accent); color: #fff; }
  .body { display: flex; flex-direction: column; gap: 0.2rem; }
  .lead { margin: 0; font-size: 0.8rem; }
  .calc { margin: 0; font-size: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: baseline; }
  .calc code { font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; font-size: 0.78rem; }
  .calc.big code { font-size: 0.92rem; }
  .hint { font-size: 0.68rem; color: var(--muted); }
  .venn-row { display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap; }
  .venn-row svg { width: 210px; height: auto; }
  .vlbl { font-size: 15px; font-weight: 700; }
  .vlbl.shared { fill: var(--fg); }
  .vcap { font-size: 9.5px; }
  .mode {
    font-weight: 500; font-size: 0.66rem; padding: 0.05rem 0.4rem;
    border-radius: 999px; margin-left: 0.35rem; vertical-align: middle;
  }
  .mode.live { background: #e1efe9; color: #0a5f4d; }
  .mode.pre { background: #f3e7d8; color: #8a4a10; }
  .steps li.skipped .lead { color: var(--muted); font-size: 0.76rem; }
  .reconcile { margin: 0.2rem 0 0; max-width: 70ch; line-height: 1.45; }
  .landed { margin: 0.15rem 0 0; font-size: 0.78rem; font-weight: 600; color: var(--accent); }
  .foot { margin: 0; font-size: 0.7rem; color: var(--muted); line-height: 1.45; max-width: 76ch; }
  @media (prefers-reduced-motion: reduce) {
    .steps li { transition: none; transform: none; }
  }
</style>
