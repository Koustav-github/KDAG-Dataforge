<script>
  // Act 3: the phase diagram. Hand-rolled inline SVG, no charting library.
  // sweep.json's points are grouped by θ; the concat arm (yKey, default
  // d_mean) is the primary series with a min-max seed error bar per point,
  // and d_avg_arm — the averaging control — is drawn as a second, muted
  // series at the same x. A θ slider snaps to whatever grid of θ values is
  // actually present in the data and dispatches `select` with the choice.
  //
  // The x axis is switchable, and defaults to θ. That is deliberate: θ is
  // where the project's one strong relationship lives (corr = −0.991 over
  // the per-θ means), and plotting only against M hid it. The M view stays
  // available because M's flatness is itself a reported finding.
  import { createEventDispatcher } from 'svelte';

  export let points = [];
  export let yKey = 'd_mean';
  export let marker = null;
  // Recomputed in the browser from the weights actually loaded, under whatever
  // merge mode and ablation set the learner currently has selected. The 33
  // precomputed points came from training and cannot move; this one can, and
  // it is the only mark on the plot that responds to the controls.
  // Shape: { theta, y, label } | null. x is derived here, from THIS component's
  // own xMode — passing a precomputed x let the caller's copy of the mode go
  // stale, which pushed the marker off-canvas and clipped its label to a
  // mysterious floating "y".
  export let livePoint = null;

  $: liveX = (() => {
    if (!livePoint) return null;
    if (xMode === 'theta') return livePoint.theta;
    const row = grouped.find((g) => closeEnough(g.theta, livePoint.theta));
    return row ? row.x : null;
  })();
  // Never draw a mark outside the plotted domain — clipped marks read as bugs.
  $: liveVisible =
    liveX !== null && liveX >= xDomain[0] && liveX <= xDomain[1] &&
    livePoint && Number.isFinite(livePoint.y);

  // 'theta' | 'm_mean'
  export let xMode = 'theta';
  $: xLabel = xMode === 'theta' ? 'designed overlap (θ)' : 'measured overlap (m_mean)';

  const dispatch = createEventDispatcher();

  const W = 760;
  const H = 400;
  const PAD = { top: 16, right: 20, bottom: 40, left: 48 };

  function mean(arr) {
    return arr.reduce((s, v) => s + v, 0) / arr.length;
  }

  function closeEnough(a, b) {
    return Math.abs(a - b) < 1e-6;
  }

  // Group by θ (rounded to avoid float-key drift), sorted ascending.
  $: grouped = (() => {
    const byTheta = new Map();
    for (const p of points) {
      const key = Math.round(p.theta * 1000) / 1000;
      if (!byTheta.has(key)) byTheta.set(key, []);
      byTheta.get(key).push(p);
    }
    return [...byTheta.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([theta, rows]) => {
        const xs = xMode === 'theta' ? [theta] : rows.map((r) => r.m_mean);
        const ys = rows.map((r) => r[yKey]);
        const avgArm = rows.map((r) => r.d_avg_arm);
        return {
          theta,
          x: mean(xs),
          y: mean(ys),
          yMin: Math.min(...ys),
          yMax: Math.max(...ys),
          avgArmY: mean(avgArm),
          avgArmMin: Math.min(...avgArm),
          avgArmMax: Math.max(...avgArm),
        };
      });
  })();

  // ---- Axis limits are reasoned, not fitted to the sample ----------------
  //
  // y is anchored at 0 always: D is merged loss minus the parent's own loss, so
  // 0 is not an empty corner of the data — it is the lossless merge, the thing
  // every point is compared against. Cropping to the data floor would
  // exaggerate differences relative to a baseline that has real meaning.
  //
  // x: theta genuinely sweeps its whole domain, so [0,1] is both honest and
  // readable. M does not: it occupies ~0.024 of [0,1], and forcing the plot to
  // the full domain crushes all 11 points into ~14px — honest but unreadable.
  // So the M view keeps a readable window AND ships the context ruler below,
  // which draws that window against the full [0,1] so the flatness stays
  // impossible to miss. Overview and detail, instead of one bad compromise.
  $: mSpan = (() => {
    if (xMode !== 'm_mean' || !grouped.length) return null;
    const xs = grouped.map((g) => g.x);
    return { lo: Math.min(...xs), hi: Math.max(...xs) };
  })();

  $: xDomain = (() => {
    if (xMode === 'theta' || !mSpan) return [0, 1];
    const pad = Math.max((mSpan.hi - mSpan.lo) * 0.35, 0.004);
    return [mSpan.lo - pad, mSpan.hi + pad];
  })();

  $: xDomainNote =
    xMode === 'theta'
      ? 'x fixed to [0, 1]: θ is a designed fraction, so 0 and 1 are its real endpoints and it sweeps all of them.'
      : `x windowed to the observed range so the points are legible — but M is a bounded [0, 1] correlation, and these 11 points span only ${mSpan ? (mSpan.hi - mSpan.lo).toFixed(3) : '—'} of it. The ruler below shows that window against the full range; read the two together, because the window alone would make a flat metric look like it sweeps the axis.`;

  $: yDomain = (() => {
    if (!grouped.length) return [0, 1];
    const ys = grouped.flatMap((g) => [g.yMin, g.yMax, g.avgArmMin, g.avgArmMax]);
    if (livePoint && Number.isFinite(livePoint.y)) ys.push(livePoint.y);
    const hi = Math.max(...ys, 0);
    // Round up to a clean step so the top gridline is a readable number.
    const step = hi > 1 ? 0.25 : 0.1;
    return [0, Math.ceil((hi * 1.08) / step) * step];
  })();
  const Y_DOMAIN_NOTE =
    'y anchored at 0: damage is merged loss minus parent loss, so 0 is a lossless merge, not an arbitrary floor.';

  function sx(v) {
    const [lo, hi] = xDomain;
    return PAD.left + ((v - lo) / (hi - lo || 1)) * (W - PAD.left - PAD.right);
  }
  function sy(v) {
    const [lo, hi] = yDomain;
    return H - PAD.bottom - ((v - lo) / (hi - lo || 1)) * (H - PAD.top - PAD.bottom);
  }

  function ticks(domain, count = 5) {
    const [lo, hi] = domain;
    const out = [];
    for (let i = 0; i <= count; i++) out.push(lo + ((hi - lo) * i) / count);
    return out;
  }

  $: xTicks = ticks(xDomain);
  $: yTicks = ticks(yDomain);

  // Sorted by x, not by θ: in the M view the θ order is not the x order, and
  // a polyline drawn in θ order visibly crosses itself.
  function pathOf(g, key) {
    return [...g]
      .sort((a, b) => a.x - b.x)
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(2)} ${sy(p[key]).toFixed(2)}`)
      .join(' ');
  }
  $: mainPath = pathOf(grouped, 'y');
  $: avgArmPath = pathOf(grouped, 'avgArmY');

  $: markerPoint = grouped.find((g) => closeEnough(g.theta, marker)) ?? null;

  $: sliderIndex = (() => {
    if (!grouped.length) return 0;
    let best = 0;
    let bestDiff = Infinity;
    grouped.forEach((g, i) => {
      const d = Math.abs(g.theta - (marker ?? g.theta));
      if (d < bestDiff) {
        bestDiff = d;
        best = i;
      }
    });
    return best;
  })();

  function onSlide(e) {
    const idx = Number(e.target.value);
    const g = grouped[idx];
    if (g) dispatch('select', g.theta);
  }
</script>

<div class="phase">
  <div class="axis-toggle" role="group" aria-label="Phase diagram x axis">
    <span class="axis-toggle-label">x axis</span>
    <button
      class:active={xMode === 'theta'}
      aria-pressed={xMode === 'theta'}
      on:click={() => (xMode = 'theta')}>θ (designed)</button>
    <button
      class:active={xMode === 'm_mean'}
      aria-pressed={xMode === 'm_mean'}
      on:click={() => (xMode = 'm_mean')}>M (measured)</button>
  </div>
  <p class="axis-note">
    {#if xMode === 'theta'}
      Against θ, damage falls near-monotonically — corr(θ, D) = −0.991 over these 11 per-θ means
      (−0.790 over all 33 individual runs). This is the project's one strong relationship.
    {:else}
      Against measured overlap M, the same damage values collapse into a band 0.024 wide:
      corr(M, D) = −0.608 over these 11 per-θ means, but +0.174 over all 33 individual runs —
      the sign flips when you stop averaging. M's flatness is the finding here.
    {/if}
  </p>
  <p class="limit-note">
    <strong>Axis limits.</strong> {xDomainNote} {Y_DOMAIN_NOTE}
  </p>



  <div class="chart-scroll">
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="Phase diagram: {xLabel} against merge damage, one point per θ with a min-max error bar across three seeds, plus the averaging control as a second series">
    <!-- axes -->
    <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} class="axis" />
    <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} class="axis" />

    {#each xTicks as t}
      <line x1={sx(t)} y1={H - PAD.bottom} x2={sx(t)} y2={H - PAD.bottom + 4} class="tick" />
      <text x={sx(t)} y={H - PAD.bottom + 16} class="tick-label" text-anchor="middle">{t.toFixed(2)}</text>
    {/each}
    {#each yTicks as t}
      <line x1={PAD.left - 4} y1={sy(t)} x2={PAD.left} y2={sy(t)} class="tick" />
      <text x={PAD.left - 8} y={sy(t) + 3} class="tick-label" text-anchor="end">{t.toFixed(2)}</text>
    {/each}

    <text x={(W + PAD.left - PAD.right) / 2} y={H - 4} class="axis-title" text-anchor="middle">{xLabel}</text>
    <text x={12} y={(H - PAD.bottom + PAD.top) / 2} class="axis-title" text-anchor="middle" transform="rotate(-90 12 {(H - PAD.bottom + PAD.top) / 2})">damage (d_mean)</text>

    <!-- muted averaging-control series, drawn first (underneath) -->
    <path d={avgArmPath} class="series series-avg" />
    {#each grouped as g}
      <line x1={sx(g.x)} y1={sy(g.avgArmMin)} x2={sx(g.x)} y2={sy(g.avgArmMax)} class="errorbar errorbar-avg" />
      <circle cx={sx(g.x)} cy={sy(g.avgArmY)} r="3" class="point point-avg" />
    {/each}

    <!-- primary concat series -->
    <path d={mainPath} class="series series-main" />
    {#each grouped as g}
      <line x1={sx(g.x)} y1={sy(g.yMin)} x2={sx(g.x)} y2={sy(g.yMax)} class="errorbar" />
      <circle cx={sx(g.x)} cy={sy(g.y)} r="4" class="point point-main" />
    {/each}

    {#if markerPoint}
      <circle cx={sx(markerPoint.x)} cy={sy(markerPoint.y)} r="7" class="marker" />
    {/if}

    <!-- The one mark that moves with the controls: damage recomputed in the
         browser under the current merge mode and ablation set. Drawn with a
         leader down to the axis so it reads as "your run", distinct from the
         precomputed series. -->
    {#if liveVisible}
      <line x1={sx(liveX)} y1={sy(livePoint.y)} x2={sx(liveX)} y2={H - PAD.bottom}
            class="live-stem" />
      <circle cx={sx(liveX)} cy={sy(livePoint.y)} r="9" class="live-halo" />
      <circle cx={sx(liveX)} cy={sy(livePoint.y)} r="5.5" class="live-dot" />
      <text x={Math.min(Math.max(sx(liveX), PAD.left + 48), W - PAD.right - 48)}
            y={sy(livePoint.y) - 15} class="live-label" text-anchor="middle">
        {livePoint.label}
      </text>
    {/if}
  </svg>
  </div>

  {#if xMode === 'm_mean' && mSpan}
    <!-- Context ruler: the plotted window drawn against M's full [0,1] domain.
         This is where the flatness lives now that the main plot is windowed. -->
    <figure class="ruler">
      <figcaption>
        Context: the window above, drawn against M's full 0–1 range
      </figcaption>
      <svg viewBox="0 0 640 46" role="img"
           aria-label={`Context ruler. Measured overlap M can range from 0 to 1. All eleven plotted points fall between ${mSpan.lo.toFixed(3)} and ${mSpan.hi.toFixed(3)}, a span of ${(mSpan.hi - mSpan.lo).toFixed(3)}.`}>
        <line x1="24" y1="26" x2="616" y2="26" class="ruler-axis" />
        {#each [0, 0.25, 0.5, 0.75, 1] as t}
          <line x1={24 + t * 592} y1="26" x2={24 + t * 592} y2="31" class="ruler-tick" />
          <text x={24 + t * 592} y="42" class="ruler-lbl" text-anchor="middle">{t}</text>
        {/each}
        <!-- the occupied sliver, with a minimum 3px width so it stays visible -->
        <rect x={24 + mSpan.lo * 592}
              width={Math.max((mSpan.hi - mSpan.lo) * 592, 3)}
              y="14" height="24" class="ruler-span" />
        <text x={24 + ((mSpan.lo + mSpan.hi) / 2) * 592} y="10"
              class="ruler-call" text-anchor="middle">
          all 11 points — {(mSpan.hi - mSpan.lo).toFixed(3)} wide
        </text>
      </svg>
    </figure>
  {/if}

  <div class="legend">
    <span class="chip main"></span> concat (d_mean)
    <span class="chip avg"></span> average (d_avg_arm) — sits above concat: averaging always hurts more
  </div>

  <div class="slider-row">
    <label for="theta-slider">θ = {grouped[sliderIndex]?.theta.toFixed(2) ?? '—'}</label>
    <input
      id="theta-slider"
      type="range"
      min="0"
      max={Math.max(grouped.length - 1, 0)}
      step="1"
      value={sliderIndex}
      on:input={onSlide}
      disabled={!grouped.length}
    />
    <div class="ticks-row">
      {#each grouped as g}
        <span>{g.theta.toFixed(1)}</span>
      {/each}
    </div>
  </div>
</div>

<style>
  .phase {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  svg {
    width: 100%;
    height: auto;
  }
  .chart-scroll { width: 100%; }
  .axis-toggle {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex-wrap: wrap;
  }
  .axis-toggle-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-right: 0.15rem;
  }
  .axis-toggle button {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--muted);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    font-size: 0.78rem;
    cursor: pointer;
  }
  .axis-toggle button.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
  }
  .axis-note {
    margin: 0;
    font-size: 0.78rem;
    color: var(--muted);
    max-width: 74ch;
    line-height: 1.45;
  }
  .limit-note {
    margin: 0;
    font-size: 0.72rem;
    color: var(--muted);
    max-width: 78ch;
    line-height: 1.4;
    border-left: 2px solid var(--border);
    padding-left: 0.55rem;
  }
  .ruler {
    margin: 0;
    display: flex;
    flex-direction: column-reverse;
    gap: 0.25rem;
  }
  .ruler figcaption {
    font-size: 0.72rem;
    color: var(--muted);
  }
  .ruler svg {
    width: 100%;
    height: auto;
    display: block;
  }
  .ruler-axis {
    stroke: var(--border);
    stroke-width: 1.5;
  }
  .ruler-tick {
    stroke: var(--border);
    stroke-width: 1;
  }
  .ruler-lbl {
    font-size: 9.5px;
    fill: var(--muted);
  }
  .ruler-span {
    fill: #c25e12;
    opacity: 0.85;
    rx: 2;
  }
  .ruler-call {
    font-size: 10px;
    font-weight: 600;
    fill: #c25e12;
  }
  .live-stem {
    stroke: #c25e12;
    stroke-width: 1.5;
    stroke-dasharray: 2 3;
    opacity: 0.55;
  }
  .live-halo {
    fill: #c25e12;
    opacity: 0.18;
  }
  .live-dot {
    fill: #c25e12;
    stroke: var(--panel-bg);
    stroke-width: 2;
  }
  .live-label {
    font-size: 10.5px;
    font-weight: 600;
    fill: #c25e12;
  }
  .axis {
    stroke: var(--border);
    stroke-width: 1;
  }
  .tick {
    stroke: var(--border);
    stroke-width: 1;
  }
  .tick-label {
    fill: var(--muted);
    font-size: 9px;
  }
  .axis-title {
    fill: var(--muted);
    font-size: 10px;
  }
  .series-main {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2;
  }
  .series-avg {
    fill: none;
    stroke: var(--muted);
    stroke-width: 1.5;
    opacity: 0.5;
    stroke-dasharray: 4 3;
  }
  .errorbar {
    stroke: var(--accent);
    stroke-width: 1.5;
    opacity: 0.55;
  }
  .errorbar-avg {
    stroke: var(--muted);
    stroke-width: 1;
    opacity: 0.35;
  }
  .point-main {
    fill: var(--accent);
  }
  .point-avg {
    fill: var(--muted);
    opacity: 0.6;
  }
  .marker {
    fill: none;
    stroke: var(--fg);
    stroke-width: 2;
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--muted);
  }
  .chip {
    display: inline-block;
    width: 14px;
    height: 3px;
    background: var(--accent);
    margin-left: 0.6rem;
  }
  .chip:first-child {
    margin-left: 0;
  }
  .chip.avg {
    background: var(--muted);
    opacity: 0.6;
  }
  .slider-row {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .slider-row label {
    font-size: 0.78rem;
    color: var(--fg);
    font-weight: 600;
  }
  input[type='range'] {
    width: 100%;
    accent-color: var(--accent);
  }
  .ticks-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: var(--muted);
    padding: 0 2px;
  }

  /* ---- Narrow screens -------------------------------------------------
     The chart's text is sized in viewBox units (9-10.5px against a 760-unit
     width). Scaling the whole SVG down to a ~310px phone viewport would
     render those labels at under 4 CSS px — unreadable. So below 640px the
     chart keeps a legible minimum width and scrolls sideways inside its own
     container instead, leaving the page itself free of horizontal scroll. */
  @media (max-width: 640px) {
    .chart-scroll {
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      /* room for the scrollbar so it never covers the x-axis labels */
      padding-bottom: 0.35rem;
    }
    .chart-scroll svg {
      min-width: 560px;
    }
    .axis-toggle {
      flex-wrap: wrap;
      gap: 0.3rem;
    }
    .ruler svg { min-width: 460px; }
    .ruler { overflow-x: auto; }
  }

  @media (pointer: coarse) {
    .axis-toggle button {
      padding: 0.45rem 0.8rem;
    }
  }
</style>
