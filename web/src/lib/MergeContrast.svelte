<script>
  // Why concatenation is available to BDH and not to a Transformer.
  //
  // THIS IS A SCHEMATIC. Nothing here is measured — no weights are loaded, no
  // forward pass runs, the bar heights are decorative. It is labelled
  // `illustration` for exactly that reason. The real, computed merge is Act 1;
  // this panel only explains why that operation is well-defined at all.
  //
  // The claim it draws, which is verified in code elsewhere in this repo:
  //   - A Transformer's parameters are pairwise-coupled (softmax over all keys,
  //     W_O mixing all heads), so merging must blend two models into one
  //     fixed-size set of weights. Capacity does not grow.
  //   - BDH's linear attention touches each neuron independently
  //     (`yKV[i] = Σ_j qr[j] * rho[j,i]`, no j-j' cross term), so two neuron
  //     populations can simply be appended — merge.js does literally
  //     `out.set(a, 0); out.set(b, a.length)`. Capacity doubles, both parents
  //     survive intact.
  import { onMount, onDestroy } from 'svelte';
  import gsap from 'gsap';
  import HonestyBadge from './HonestyBadge.svelte';

  const A_COLOR = '#0a8f72';
  const B_COLOR = '#c25e12';
  const MUD = '#8b8476';

  // Transformer panel: two stacks of independently-parameterised layers.
  const TF_LAYERS = 4;
  const TF_W = 76;
  const TF_H = 12;
  const TF_GAP = 5;
  const TF_TOP = 62;
  const TF_AX = 60;                 // model A resting x
  const TF_BX = 214;                // model B resting x
  const TF_CX = 137;                // where both must land: the SAME slot
  const tfRows = Array.from({ length: TF_LAYERS }, (_, i) => TF_TOP + i * (TF_H + TF_GAP));

  // BDH panel: two neuron populations, drawn as rows of units.
  const N_BARS = 10;
  const BAR_W = 6;
  const BAR_GAP = 3;
  const ROW_W = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP;   // 87
  const BDH_AX = 430;
  const BDH_BX = 573;
  const PAIR_W = ROW_W * 2 + BAR_GAP;                   // 177 once appended
  const BDH_CX = 545 - PAIR_W / 2;                      // 456.5
  const BDH_A_TO = BDH_CX;
  const BDH_B_TO = BDH_CX + ROW_W + BAR_GAP;

  // Decorative, deterministic bar heights — NOT model magnitudes. A fixed
  // pattern rather than Math.random() so the picture is identical on every
  // load and cannot be mistaken for data that varies run to run.
  const heights = (offset) =>
    Array.from({ length: N_BARS }, (_, i) => 12 + ((i * 7 + offset * 5) % 19));
  const hA = heights(0);
  const hB = heights(3);
  const BAR_BASE = 112;             // baseline the bars sit on

  let tfA, tfB, tfMerged, tfNote;
  let bdhA, bdhB, bdhBrace, bdhNote;
  let tl;
  let phase = 'two trained models';
  let paused = false;

  const reduced = () =>
    typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  function settle() {
    // Final state, no motion — used when the viewer prefers reduced motion.
    gsap.set([tfA, tfB], { x: TF_CX - TF_AX, opacity: 0 });
    gsap.set(tfB, { x: TF_CX - TF_BX });
    gsap.set([tfMerged, tfNote, bdhBrace, bdhNote], { opacity: 1 });
    gsap.set(bdhA, { x: BDH_A_TO - BDH_AX });
    gsap.set(bdhB, { x: BDH_B_TO - BDH_BX });
    phase = 'merged';
  }

  function build() {
    tl = gsap.timeline({ repeat: -1, repeatDelay: 1.1, defaults: { ease: 'power2.inOut' } });

    // 0.0 — rest. Two separate models on each side.
    tl.set([tfA, tfB], { x: 0, opacity: 1 })
      .set(tfMerged, { opacity: 0 })
      .set([tfNote, bdhBrace, bdhNote], { opacity: 0 })
      .set([bdhA, bdhB], { x: 0 })
      .call(() => (phase = 'two trained models'));

    // 0.6 — both sides move together.
    tl.call(() => (phase = 'merge'), null, 0.6)
      .to(tfA, { x: TF_CX - TF_AX, duration: 1.0 }, 0.6)
      .to(tfB, { x: TF_CX - TF_BX, duration: 1.0 }, 0.6)
      .to(bdhA, { x: BDH_A_TO - BDH_AX, duration: 1.0 }, 0.6)
      .to(bdhB, { x: BDH_B_TO - BDH_BX, duration: 1.0 }, 0.6);

    // 1.6 — the divergence. Transformer: the two collapse into one fixed-size
    // set of weights and lose their identity. BDH: nothing collapses; the two
    // populations simply sit side by side and the axis is now twice as long.
    tl.to([tfA, tfB], { opacity: 0, duration: 0.55 }, 1.6)
      .to(tfMerged, { opacity: 1, duration: 0.55 }, 1.6)
      .to(bdhBrace, { opacity: 1, duration: 0.55 }, 1.7)
      .call(() => (phase = 'result'), null, 2.2);

    // 2.3 — the captions that say what just happened.
    tl.to([tfNote, bdhNote], { opacity: 1, duration: 0.5 }, 2.3);

    // hold, then fade the result so the loop restart is not a hard cut.
    tl.to({}, { duration: 2.0 })
      .to([tfMerged, tfNote, bdhBrace, bdhNote], { opacity: 0, duration: 0.5 });
  }

  onMount(() => {
    if (reduced()) { settle(); return; }
    build();
    return () => tl && tl.kill();
  });

  onDestroy(() => { if (tl) tl.kill(); });

  function toggle() {
    if (!tl) return;
    paused = !paused;
    paused ? tl.pause() : tl.play();
  }
</script>

<div class="contrast">
  <div class="head">
    <span class="title">Why this merge is a BDH move, not a general one</span>
    <span class="head-right">
      <HonestyBadge kind="illustration" />
      {#if tl}
        <button class="toggle" on:click={toggle} aria-label={paused ? 'Play animation' : 'Pause animation'}>
          {paused ? '▶ play' : '❚❚ pause'}
        </button>
      {/if}
    </span>
  </div>

  <div class="stage">
    <svg viewBox="0 0 720 220" role="img"
         aria-label="Schematic comparing two merge operations. On the left, two Transformers merge by averaging into a single fixed-size set of weights, so capacity does not grow and neither original survives. On the right, two BDH models merge by appending their neuron populations, so the neuron axis doubles and both parents remain intact.">

      <!-- ── divider ─────────────────────────────────────────────── -->
      <line x1="360" y1="14" x2="360" y2="206" class="divider" />

      <!-- ══ LEFT: Transformer ═══════════════════════════════════════ -->
      <text x="175" y="22" class="panel-title" text-anchor="middle">Transformer</text>
      <text x="175" y="40" class="panel-sub" text-anchor="middle">weights are pairwise-coupled</text>

      <!-- model A -->
      <g bind:this={tfA}>
        {#each tfRows as y}
          <rect x={TF_AX} {y} width={TF_W} height={TF_H} rx="2" fill={A_COLOR} opacity="0.85" />
        {/each}
        <text x={TF_AX + TF_W / 2} y={TF_TOP - 8} class="glyph-label" text-anchor="middle" fill={A_COLOR}>A</text>
      </g>

      <!-- model B -->
      <g bind:this={tfB}>
        {#each tfRows as y}
          <rect x={TF_BX} {y} width={TF_W} height={TF_H} rx="2" fill={B_COLOR} opacity="0.85" />
        {/each}
        <text x={TF_BX + TF_W / 2} y={TF_TOP - 8} class="glyph-label" text-anchor="middle" fill={B_COLOR}>B</text>
      </g>

      <!-- the averaged result: same slot, same width, neither parent legible -->
      <g bind:this={tfMerged}>
        {#each tfRows as y}
          <rect x={TF_CX} {y} width={TF_W} height={TF_H} rx="2" fill={MUD} opacity="0.9" />
        {/each}
        <text x={TF_CX + TF_W / 2} y={TF_TOP - 8} class="glyph-label" text-anchor="middle" fill={MUD}>
          (A+B)/2
        </text>
      </g>

      <!-- width marker: unchanged -->
      <g bind:this={tfNote}>
        <line x1={TF_CX} y1="140" x2={TF_CX + TF_W} y2="140" class="brace" />
        <line x1={TF_CX} y1="136" x2={TF_CX} y2="144" class="brace" />
        <line x1={TF_CX + TF_W} y1="136" x2={TF_CX + TF_W} y2="144" class="brace" />
        <text x={TF_CX + TF_W / 2} y="158" class="measure" text-anchor="middle">same size</text>
        <text x="175" y="180" class="why" text-anchor="middle">softmax couples every key;</text>
        <text x="175" y="194" class="why" text-anchor="middle">W&#8338; mixes every head — no row to append</text>
      </g>

      <!-- ══ RIGHT: BDH ══════════════════════════════════════════════ -->
      <text x="545" y="22" class="panel-title" text-anchor="middle">BDH</text>
      <text x="545" y="40" class="panel-sub" text-anchor="middle">each neuron is independent</text>

      <!-- neuron population A -->
      <g bind:this={bdhA}>
        {#each hA as h, i}
          <rect x={BDH_AX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h}
                width={BAR_W} height={h} rx="1.5" fill={A_COLOR} opacity="0.9" />
        {/each}
        <text x={BDH_AX + ROW_W / 2} y="54" class="glyph-label" text-anchor="middle" fill={A_COLOR}>A</text>
      </g>

      <!-- neuron population B -->
      <g bind:this={bdhB}>
        {#each hB as h, i}
          <rect x={BDH_BX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h}
                width={BAR_W} height={h} rx="1.5" fill={B_COLOR} opacity="0.9" />
        {/each}
        <text x={BDH_BX + ROW_W / 2} y="54" class="glyph-label" text-anchor="middle" fill={B_COLOR}>B</text>
      </g>

      <!-- width marker: doubled, and both halves still named -->
      <g bind:this={bdhBrace}>
        <line x1={BDH_CX} y1="126" x2={BDH_CX + PAIR_W} y2="126" class="brace" />
        <line x1={BDH_CX} y1="122" x2={BDH_CX} y2="130" class="brace" />
        <line x1={BDH_CX + PAIR_W} y1="122" x2={BDH_CX + PAIR_W} y2="130" class="brace" />
        <line x1={BDH_CX + ROW_W + BAR_GAP / 2} y1="70" x2={BDH_CX + ROW_W + BAR_GAP / 2} y2="122"
              class="seam" />
        <text x={BDH_CX + PAIR_W / 2} y="144" class="measure" text-anchor="middle">n doubled</text>
      </g>

      <g bind:this={bdhNote}>
        <text x={BDH_CX + ROW_W / 2} y="158" class="measure" text-anchor="middle" fill={A_COLOR}>A intact</text>
        <text x={BDH_CX + ROW_W + BAR_GAP + ROW_W / 2} y="158" class="measure" text-anchor="middle" fill={B_COLOR}>B intact</text>
        <text x="545" y="180" class="why" text-anchor="middle">linear attention touches one neuron at a time —</text>
        <text x="545" y="194" class="why" text-anchor="middle">so the populations simply append</text>
      </g>
    </svg>
  </div>

  <p class="read">
    <strong>{phase === 'result' ? 'The difference:' : 'Watching:'}</strong>
    a Transformer merge has to fit two models into one fixed-size set of weights, so the
    parents blend and capacity stays flat. BDH appends one neuron population to the other:
    the axis grows to 2n and both parents survive as themselves.
  </p>
  <p class="read scope">
    <strong>What is and isn't measured here.</strong> The BDH side draws a real operation —
    Act 1 below runs exactly this concatenation on loaded weights, and every number in this
    artifact comes from it. The Transformer side is a structural claim, not an experiment we
    ran: this project contains no Transformer. It follows from the architecture — softmax
    normalises over all keys jointly and W&#8338; mixes all heads, so there is no independent
    row to append — which is why the field's Transformer merges average weights
    (arXiv:2203.05482) or add task vectors (arXiv:2212.04089) instead of growing capacity.
  </p>
</div>

<style>
  .contrast {
    border: 1px dashed #c9a84c;
    background: #fbf6e6;
    border-radius: 10px;
    padding: 0.75rem 0.9rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .head-right { display: inline-flex; align-items: center; gap: 0.4rem; }
  .title { font-size: 0.83rem; font-weight: 600; color: var(--fg); }
  .toggle {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--muted);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.68rem;
    cursor: pointer;
    font-family: inherit;
  }
  .stage { width: 100%; }
  svg { width: 100%; height: auto; display: block; }

  .divider { stroke: #e2d9bd; stroke-width: 1; }
  .panel-title { font-size: 12.5px; font-weight: 700; fill: #1c1b19; font-family: inherit; }
  .panel-sub { font-size: 9.5px; fill: #8a8474; font-family: inherit; }
  .glyph-label { font-size: 10px; font-weight: 700; font-family: inherit; }
  .brace { stroke: #6b6459; stroke-width: 1.2; }
  .seam { stroke: #1c1b19; stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.55; }
  .measure { font-size: 9.5px; font-weight: 600; fill: #6b6459; font-family: inherit; }
  .why { font-size: 9px; fill: #8a8474; font-family: inherit; }

  .read {
    margin: 0;
    font-size: 0.76rem;
    line-height: 1.5;
    color: var(--muted);
    max-width: 78ch;
  }
  .read strong { color: var(--fg); }
  .scope {
    font-size: 0.71rem;
    border-left: 2px solid #c9a84c;
    padding-left: 0.55rem;
  }

  /* Wide schematic: keep the text legible on a phone rather than scaling the
     whole viewBox down to unreadable, matching the other charts' behaviour. */
  @media (max-width: 640px) {
    .stage { overflow-x: auto; -webkit-overflow-scrolling: touch; padding-bottom: 0.3rem; }
    .stage svg { min-width: 560px; }
  }
</style>
