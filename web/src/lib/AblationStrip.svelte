<script>
  // Act 4's headline, as a picture rather than four numbers.
  //
  // The question is positional, and the layout exists to make exactly one
  // comparison easy: does the collision marker sit outside the band of random
  // draws, or inside it? Inside means removing the neurons we picked is
  // indistinguishable from removing that many at random — which is what we
  // actually measure. Everything else on the chart is there to make that
  // comparison legible: the baseline says where we started, the band says what
  // chance looks like, and the gap arrow says how far from chance we landed.

  export let baseline = null;    // loss with nothing ablated
  export let collision = null;   // loss with the top-k collision set ablated
  export let draws = [];         // one loss per random draw, same k
  export let mean = null;        // mean of draws
  export let sd = null;          // sample SD of draws

  const COLLISION = '#0a8f72';
  const RANDOM = '#c25e12';
  const INK = '#1c1b19';
  const MUTED = '#726f68';

  const W = 720;
  const H = 240;
  const PADL = 30;
  const PADR = 30;
  // Vertical bands, top to bottom, each with its own clearance so no two
  // labels can ever land on each other or on the axis:
  //   30  baseline label
  //   54  random / collision labels
  //   96  the marks (band spans 66..126)
  //  140  gap-arrow label · 146 the arrow itself
  //  168  "what chance does" caption
  //  188  axis · 205 tick values · 232 footnotes
  const TRACK = 96;
  const AXIS = 188;

  $: ready = baseline !== null && collision !== null && mean !== null && draws.length > 1;

  $: values = ready ? [baseline, collision, ...draws] : [];
  $: lo = ready ? Math.min(...values) : 0;
  $: hi = ready ? Math.max(...values) : 1;
  $: pad = ready ? Math.max((hi - lo) * 0.28, 0.004) : 0.1;
  $: d0 = lo - pad;
  $: d1 = hi + pad;
  $: x = (v) => PADL + ((v - d0) / (d1 - d0)) * (W - PADL - PADR);
  // keep any label fully inside the frame
  $: clamp = (px, half) => Math.min(Math.max(px, PADL + half), W - PADR - half);

  $: bandLo = ready && sd !== null ? mean - sd : null;
  $: bandHi = ready && sd !== null ? mean + sd : null;

  // The single fact the picture must carry.
  $: inside = ready && sd !== null && collision >= bandLo && collision <= bandHi;
  $: gap = ready ? mean - collision : null;         // >0 means collision beat chance
  $: sigmas = ready && sd ? Math.abs(gap) / sd : null;

  $: verdict = !ready
    ? ''
    : inside
      ? 'inside the band — no better than chance'
      : gap > 0
        ? 'left of the band — genuinely beat chance'
        : 'right of the band — worse than chance';

  $: ticks = ready ? [d0 + (d1 - d0) * 0.12, (d0 + d1) / 2, d1 - (d1 - d0) * 0.12] : [];

  const f = (v, p = 3) => (v === null || Number.isNaN(v) ? '—' : v.toFixed(p));
</script>

{#if !ready}
  <p class="waiting">computing both arms…</p>
{:else}
  <figure class="strip">
    <figcaption class="question">
      Did ablating the collision set beat removing the same number of neurons at random?
    </figcaption>

    <div class="chart-scroll">
    <svg viewBox="0 0 {W} {H}" role="img"
         aria-label={`Baseline loss ${f(baseline)}. Ablating the collision set gives ${f(collision)}. Ablating ${draws.length} random sets of the same size averages ${f(mean)} with standard deviation ${f(sd)}, a noise band from ${f(bandLo)} to ${f(bandHi)}. The collision result sits ${verdict}.`}>

      <!-- ── what chance looks like ───────────────────────────────── -->
      <rect x={x(bandLo)} y={TRACK - 30} width={Math.max(x(bandHi) - x(bandLo), 2)} height="60"
            fill={RANDOM} opacity="0.12" rx="3" />
      <line x1={x(bandLo)} y1={TRACK - 30} x2={x(bandLo)} y2={TRACK + 30}
            stroke={RANDOM} stroke-width="2" opacity="0.7" />
      <line x1={x(bandHi)} y1={TRACK - 30} x2={x(bandHi)} y2={TRACK + 30}
            stroke={RANDOM} stroke-width="2" opacity="0.7" />
      <text x={clamp((x(bandLo) + x(bandHi)) / 2, 90)} y={TRACK + 72}
            class="zone" text-anchor="middle" fill={RANDOM}>
        what chance does — {draws.length} random ablations, ±1 SD
      </text>

      <!-- each individual random draw -->
      {#each draws as dv}
        <line x1={x(dv)} y1={TRACK - 12} x2={x(dv)} y2={TRACK + 12}
              stroke={RANDOM} stroke-width="2" opacity="0.45" />
      {/each}
      <circle cx={x(mean)} cy={TRACK} r="5" fill={RANDOM} stroke="#f7f6f3" stroke-width="2" />
      <text x={clamp(x(mean), 56)} y={TRACK - 42} class="lbl" text-anchor="middle" fill={RANDOM}>
        random avg {f(mean)}
      </text>

      <!-- ── where we started ─────────────────────────────────────── -->
      <line x1={x(baseline)} y1={TRACK - 30} x2={x(baseline)} y2={AXIS}
            stroke={MUTED} stroke-width="1.5" stroke-dasharray="3 3" />
      <text x={clamp(x(baseline), 52)} y={TRACK - 66} class="lbl muted" text-anchor="middle">
        before ablation {f(baseline)}
      </text>

      <!-- ── the result under test ────────────────────────────────── -->
      <circle cx={x(collision)} cy={TRACK} r="8.5" fill={COLLISION}
              stroke="#f7f6f3" stroke-width="2.5" />
      <text x={clamp(x(collision), 60)} y={TRACK - 42} class="lbl strong"
            text-anchor="middle" fill={COLLISION}>
        collision set {f(collision)}
      </text>

      <!-- ── the headline: how far from chance ────────────────────── -->
      {#if Math.abs(x(collision) - x(mean)) > 14}
        <line x1={x(mean)} y1={TRACK + 50} x2={x(collision)} y2={TRACK + 50}
              stroke={INK} stroke-width="1.2" />
        <line x1={x(mean)} y1={TRACK + 45} x2={x(mean)} y2={TRACK + 55} stroke={INK} stroke-width="1.2" />
        <line x1={x(collision)} y1={TRACK + 45} x2={x(collision)} y2={TRACK + 55} stroke={INK} stroke-width="1.2" />
        <text x={clamp((x(mean) + x(collision)) / 2, 74)} y={TRACK + 38}
              class="gap" text-anchor="middle">
          {gap >= 0 ? '+' : '−'}{f(Math.abs(gap))} vs chance{sigmas ? ` · ${sigmas.toFixed(1)}σ` : ''}
        </text>
      {/if}

      <!-- ── axis ─────────────────────────────────────────────────── -->
      <line x1={PADL} y1={AXIS} x2={W - PADR} y2={AXIS} stroke="#ddd9d1" stroke-width="1" />
      {#each ticks as t}
        <line x1={x(t)} y1={AXIS} x2={x(t)} y2={AXIS + 4} stroke="#ddd9d1" stroke-width="1" />
        <text x={x(t)} y={AXIS + 17} class="tick" text-anchor="middle">{t.toFixed(3)}</text>
      {/each}
      <text x={PADL} y={H - 6} class="tick" text-anchor="start">← lower loss is better</text>
      <text x={W - PADR} y={H - 6} class="tick" text-anchor="end">mean cross-entropy</text>
    </svg>
    </div>

    <p class="read" class:bad={inside || gap < 0}>
      <strong>Read it like this.</strong>
      The shaded band is what happens when you ablate {draws.length} <em>random</em> sets of the same
      size — that is the noise floor, and anything landing inside it is not evidence of anything.
      {#if inside}
        The teal marker is <strong>inside</strong> that band, so the neurons our collision metric
        picked are <strong>no better than a blind guess</strong> at this k.
      {:else if gap < 0}
        The teal marker sits <strong>to the right</strong> of it — ablating the neurons we picked
        left the model <strong>{f(Math.abs(gap))} worse</strong> than removing random ones
        ({sigmas ? sigmas.toFixed(1) : '—'}σ). Not a localization effect; the opposite of one.
      {:else}
        The teal marker sits <strong>to the left</strong> of it — the neurons we picked genuinely
        beat chance by {f(gap)} ({sigmas ? sigmas.toFixed(1) : '—'}σ).
      {/if}
      The raw drop from “before ablation” is not the result; the distance from the band is.
    </p>

    <div class="key">
      <span><i class="dot" style="background:{COLLISION}"></i>collision set</span>
      <span><i class="dot" style="background:{RANDOM}"></i>random draws + their average</span>
      <span><i class="band" style="background:{RANDOM}"></i>±1 SD noise band</span>
      <span><i class="dash"></i>before ablation</span>
    </div>
  </figure>
{/if}

<style>
  .strip { margin: 0; display: flex; flex-direction: column; gap: 0.55rem; }
  svg { width: 100%; height: auto; display: block; }
  .chart-scroll { width: 100%; }

  /* Text here is sized in viewBox units against a 720-unit width, so a phone
     viewport would shrink the 10.5px ticks to well under 5 CSS px. Keep a
     legible floor and scroll within the figure instead. */
  @media (max-width: 640px) {
    .chart-scroll {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      padding-bottom: 0.35rem;
    }
    .chart-scroll svg { min-width: 520px; }
  }
  .question { font-size: 0.85rem; font-weight: 600; color: var(--fg); }
  .lbl { font-size: 12.5px; font-family: inherit; }
  .lbl.strong { font-weight: 700; }
  .lbl.muted { fill: var(--muted); }
  .zone { font-size: 11px; font-family: inherit; }
  .gap { font-size: 12px; font-weight: 700; fill: #1c1b19; font-family: inherit; }
  .tick { font-size: 10.5px; fill: var(--muted); font-family: inherit; }
  .read {
    margin: 0;
    font-size: 0.8rem;
    line-height: 1.55;
    border-left: 3px solid var(--accent);
    padding-left: 0.65rem;
    max-width: 78ch;
  }
  .read.bad { border-left-color: #c25e12; }
  .key { display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.72rem; color: var(--muted); }
  .key span { display: inline-flex; align-items: center; gap: 0.3rem; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .band { width: 14px; height: 9px; border-radius: 2px; opacity: 0.22; display: inline-block; }
  .dash { width: 14px; height: 0; display: inline-block; border-top: 2px dashed var(--muted); }
  .waiting { color: var(--muted); font-size: 0.85rem; }
</style>
