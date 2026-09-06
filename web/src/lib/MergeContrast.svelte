<script>
  // Three panels: why a Transformer cannot concatenate, and the two rules BDH
  // actually offers.
  //
  // THIS IS A SCHEMATIC — no weights are loaded and no forward pass runs here.
  // Panels 2 and 3 draw operations that ARE real: mergeAverage and mergeConcat
  // run in the browser on the loaded parents in Acts 1-2, and the comparison
  // between them is this project's most replicated result. Panel 1 is a
  // structural claim, not an experiment: there is no Transformer in this repo.
  //
  //   1. Transformer, attempting concatenation
  //      The heads append fine. W_O does not: it maps ALL heads back to d_model
  //      through one dense matrix, so doubling the heads needs a twice-as-wide
  //      W_O whose cross-terms exist in neither parent. Fall back to averaging.
  //
  //   2. BDH, average  (merge.js:blend, concat=false)
  //      Every tensor elementwise-averaged, n stays n. The same operation a
  //      Transformer is limited to. In this project it is the CONTROL arm.
  //
  //   3. BDH, concatenate  (merge.js:blend, concat=true)
  //      Neuron-axis tensors appended (n -> 2n), the rest averaged. Available
  //      because linear attention touches one neuron at a time
  //      (yKV[i] = sum_j qr[j]*rho[j,i], no j-j' cross term).
  //
  // Each panel is its own SVG in its own local coordinate system, so the three
  // can be laid out in a row or a column by CSS alone — on a narrow screen they
  // stack full-width instead of shrinking their labels to nothing.
  import { onMount, onDestroy } from 'svelte';
  import gsap from 'gsap';
  import HonestyBadge from './HonestyBadge.svelte';

  const A_COLOR = '#0a8f72';
  const B_COLOR = '#c25e12';
  const MUD = '#8b8476';
  const BAD = '#a5453b';

  // One panel's local frame. Every panel uses the same centre.
  const PW = 290, PH = 214, C = PW / 2;

  // ── Transformer glyph: a row of heads over the W_O mixing matrix ──
  const HEAD_W = 14, HEAD_H = 14, HEAD_GAP = 3, N_HEADS = 3;
  const HEADS_W = N_HEADS * (HEAD_W + HEAD_GAP) - HEAD_GAP;   // 48
  const TF_PAIR = HEADS_W * 2 + HEAD_GAP;                     // 99
  const OFF = 24;
  const tfAX = C - HEADS_W - OFF, tfBX = C + OFF;
  const tfATo = C - TF_PAIR / 2, tfBTo = tfATo + HEADS_W + HEAD_GAP;
  const HEAD_Y = 72, WO_Y = 94, WO_H = 11;

  // ── BDH glyph: a row of independent neurons ──────────────────────
  const N_BARS = 8, BAR_W = 6, BAR_GAP = 3;
  const ROW_W = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP;         // 69
  const PAIR_W = ROW_W * 2 + BAR_GAP;                         // 141
  const BAR_BASE = 118;

  const barAX = C - ROW_W - OFF, barBX = C + OFF;
  const avgTo = C - ROW_W / 2;
  const catATo = C - PAIR_W / 2, catBTo = catATo + ROW_W + BAR_GAP;

  // Decorative, deterministic heights — NOT model magnitudes. Fixed pattern
  // rather than Math.random(), so the picture is identical on every load and
  // cannot be mistaken for data that varies run to run.
  const heights = (o) => Array.from({ length: N_BARS }, (_, i) => 14 + ((i * 7 + o * 5) % 20));
  const hA = heights(0), hB = heights(3);
  const hMud = hA.map((h, i) => (h + hB[i]) / 2);

  let tfA, tfB, tfFail, tfNote;
  let avgA, avgB, avgMud, avgNote;
  let catA, catB, catBrace, catNote;
  let tl, paused = false;

  const reduced = () =>
    typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  function settle() {
    gsap.set(tfA, { x: tfATo - tfAX });
    gsap.set(tfB, { x: tfBTo - tfBX });
    gsap.set(avgA, { x: avgTo - barAX, opacity: 0 });
    gsap.set(avgB, { x: avgTo - barBX, opacity: 0 });
    gsap.set(catA, { x: catATo - barAX });
    gsap.set(catB, { x: catBTo - barBX });
    gsap.set([tfFail, tfNote, avgMud, avgNote, catBrace, catNote], { opacity: 1 });
  }

  function build() {
    tl = gsap.timeline({ repeat: -1, repeatDelay: 1.2, defaults: { ease: 'power2.inOut' } });

    tl.set([tfA, tfB, avgA, avgB, catA, catB], { x: 0, opacity: 1 })
      .set([tfFail, tfNote, avgMud, avgNote, catBrace, catNote], { opacity: 0 });

    // All three panels attempt the same move at the same time.
    tl.to(tfA, { x: tfATo - tfAX, duration: 1.0 }, 0.6)
      .to(tfB, { x: tfBTo - tfBX, duration: 1.0 }, 0.6)
      .to(avgA, { x: avgTo - barAX, duration: 1.0 }, 0.6)
      .to(avgB, { x: avgTo - barBX, duration: 1.0 }, 0.6)
      .to(catA, { x: catATo - barAX, duration: 1.0 }, 0.6)
      .to(catB, { x: catBTo - barBX, duration: 1.0 }, 0.6);

    // Where the three diverge.
    tl.to(tfFail, { opacity: 1, duration: 0.5 }, 1.65)       // W_O has no valid shape
      .to([avgA, avgB], { opacity: 0, duration: 0.55 }, 1.6)  // parents blend away
      .to(avgMud, { opacity: 1, duration: 0.55 }, 1.6)
      .to(catBrace, { opacity: 1, duration: 0.55 }, 1.7);     // axis simply grew

    tl.to([tfNote, avgNote, catNote], { opacity: 1, duration: 0.5 }, 2.35);

    tl.to({}, { duration: 2.2 })
      .to([tfFail, tfNote, avgMud, avgNote, catBrace, catNote], { opacity: 0, duration: 0.5 });
  }

  onMount(() => { if (reduced()) settle(); else build(); });
  onDestroy(() => { if (tl) tl.kill(); });

  function toggle() { if (!tl) return; paused = !paused; paused ? tl.pause() : tl.play(); }
</script>

<div class="contrast">
  <div class="head">
    <span class="title">Why concatenation is a BDH move — and how it compares to averaging</span>
    <span class="head-right">
      <HonestyBadge kind="illustration" />
      {#if tl}
        <button class="toggle" on:click={toggle}
                aria-label={paused ? 'Play animation' : 'Pause animation'}>
          {paused ? '▶ play' : '❚❚ pause'}
        </button>
      {/if}
    </span>
  </div>

  <div class="stage">
    <!-- ══ 1 · TRANSFORMER: concatenation unavailable ═══════════════ -->
    <div class="panel">
      <svg viewBox="0 0 {PW} {PH}" role="img"
           aria-label="Two Transformers try to concatenate their attention heads. The heads append, but the output mixing matrix W-O would need to be twice as wide and those weights exist in neither parent, so concatenation is unavailable and the models must be averaged instead.">
        <text x={C} y="22" class="panel-title" text-anchor="middle">Transformer</text>
        <text x={C} y="37" class="panel-sub" text-anchor="middle">tries to concatenate</text>

        <g bind:this={tfA}>
          {#each Array(N_HEADS) as _, i}
            <rect x={tfAX + i * (HEAD_W + HEAD_GAP)} y={HEAD_Y} width={HEAD_W} height={HEAD_H}
                  rx="2" fill={A_COLOR} opacity="0.9" />
          {/each}
          <rect x={tfAX} y={WO_Y} width={HEADS_W} height={WO_H} rx="2" fill={A_COLOR} opacity="0.45" />
          <text x={tfAX + HEADS_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={A_COLOR}>A</text>
        </g>

        <g bind:this={tfB}>
          {#each Array(N_HEADS) as _, i}
            <rect x={tfBX + i * (HEAD_W + HEAD_GAP)} y={HEAD_Y} width={HEAD_W} height={HEAD_H}
                  rx="2" fill={B_COLOR} opacity="0.9" />
          {/each}
          <rect x={tfBX} y={WO_Y} width={HEADS_W} height={WO_H} rx="2" fill={B_COLOR} opacity="0.45" />
          <text x={tfBX + HEADS_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={B_COLOR}>B</text>
        </g>

        <!-- heads append fine; W_O is what breaks -->
        <g bind:this={tfFail}>
          <rect x={tfATo} y={WO_Y - 2} width={TF_PAIR} height={WO_H + 4} rx="3"
                fill="none" stroke={BAD} stroke-width="1.6" stroke-dasharray="4 3" />
          <text x={C} y={WO_Y + WO_H - 1} class="fail-mark" text-anchor="middle">?</text>
          <text x={C} y="130" class="fail-note" text-anchor="middle">W&#8338; must be twice as wide</text>
          <text x={C} y="143" class="fail-note" text-anchor="middle">those weights exist in neither parent</text>
        </g>

        <g bind:this={tfNote}>
          <text x={C} y="168" class="why" text-anchor="middle">softmax couples every key;</text>
          <text x={C} y="181" class="why" text-anchor="middle">W&#8338; mixes every head — no free axis</text>
          <text x={C} y="201" class="verdict" text-anchor="middle" fill={BAD}>✗ must average instead</text>
        </g>
      </svg>
    </div>

    <!-- ══ 2 · BDH: average ════════════════════════════════════════ -->
    <div class="panel">
      <svg viewBox="0 0 {PW} {PH}" role="img"
           aria-label="Two BDH models merged by averaging: both parents land in one fixed-size set of weights and blend together, so the neuron count is unchanged and neither parent survives distinctly.">
        <text x={C} y="22" class="panel-title" text-anchor="middle">BDH · average</text>
        <text x={C} y="37" class="panel-sub" text-anchor="middle">the same fallback, on BDH</text>

        <g bind:this={avgA}>
          {#each hA as h, i}
            <rect x={barAX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h} width={BAR_W} height={h}
                  rx="1.5" fill={A_COLOR} opacity="0.9" />
          {/each}
          <text x={barAX + ROW_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={A_COLOR}>A</text>
        </g>
        <g bind:this={avgB}>
          {#each hB as h, i}
            <rect x={barBX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h} width={BAR_W} height={h}
                  rx="1.5" fill={B_COLOR} opacity="0.9" />
          {/each}
          <text x={barBX + ROW_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={B_COLOR}>B</text>
        </g>
        <g bind:this={avgMud}>
          {#each hMud as h, i}
            <rect x={avgTo + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h} width={BAR_W} height={h}
                  rx="1.5" fill={MUD} opacity="0.95" />
          {/each}
          <text x={avgTo + ROW_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={MUD}>(A+B)/2</text>
        </g>
        <g bind:this={avgNote}>
          <line x1={avgTo} y1="128" x2={avgTo + ROW_W} y2="128" class="brace" />
          <line x1={avgTo} y1="124" x2={avgTo} y2="132" class="brace" />
          <line x1={avgTo + ROW_W} y1="124" x2={avgTo + ROW_W} y2="132" class="brace" />
          <text x={C} y="146" class="measure" text-anchor="middle">n unchanged</text>
          <text x={C} y="168" class="why" text-anchor="middle">both parents blended into one</text>
          <text x={C} y="181" class="why" text-anchor="middle">fixed-size set of weights</text>
          <text x={C} y="201" class="verdict" text-anchor="middle" fill={MUD}>control arm</text>
        </g>
      </svg>
    </div>

    <!-- ══ 3 · BDH: concatenate ════════════════════════════════════ -->
    <div class="panel">
      <svg viewBox="0 0 {PW} {PH}" role="img"
           aria-label="Two BDH models merged by concatenation: one neuron population is appended to the other, so the neuron count doubles and both parents stay intact.">
        <text x={C} y="22" class="panel-title" text-anchor="middle">BDH · concatenate</text>
        <text x={C} y="37" class="panel-sub" text-anchor="middle">what the neuron axis allows</text>

        <g bind:this={catA}>
          {#each hA as h, i}
            <rect x={barAX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h} width={BAR_W} height={h}
                  rx="1.5" fill={A_COLOR} opacity="0.9" />
          {/each}
          <text x={barAX + ROW_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={A_COLOR}>A</text>
        </g>
        <g bind:this={catB}>
          {#each hB as h, i}
            <rect x={barBX + i * (BAR_W + BAR_GAP)} y={BAR_BASE - h} width={BAR_W} height={h}
                  rx="1.5" fill={B_COLOR} opacity="0.9" />
          {/each}
          <text x={barBX + ROW_W / 2} y="60" class="glyph-label" text-anchor="middle" fill={B_COLOR}>B</text>
        </g>
        <g bind:this={catBrace}>
          <line x1={catBTo - BAR_GAP / 2} y1="74" x2={catBTo - BAR_GAP / 2} y2="124" class="seam" />
          <line x1={catATo} y1="128" x2={catATo + PAIR_W} y2="128" class="brace" />
          <line x1={catATo} y1="124" x2={catATo} y2="132" class="brace" />
          <line x1={catATo + PAIR_W} y1="124" x2={catATo + PAIR_W} y2="132" class="brace" />
          <text x={C} y="146" class="measure" text-anchor="middle">n doubled</text>
        </g>
        <g bind:this={catNote}>
          <text x={catATo + ROW_W / 2} y="168" class="why" text-anchor="middle" fill={A_COLOR}>A intact</text>
          <text x={catBTo + ROW_W / 2} y="168" class="why" text-anchor="middle" fill={B_COLOR}>B intact</text>
          <text x={C} y="181" class="why" text-anchor="middle">neurons are independent — just append</text>
          <text x={C} y="201" class="verdict" text-anchor="middle" fill={A_COLOR}>✓ wins 33/33 runs</text>
        </g>
      </svg>
    </div>
  </div>

  <p class="read">
    <strong>Left is why, middle and right are what we measured.</strong>
    A Transformer's heads would append happily — it is W&#8338;, the matrix that mixes every
    head back down, that has no valid wider form. So it must average. BDH can average too
    (middle): that is exactly the control we compare against. What BDH additionally allows
    is concatenation (right), because linear attention touches one neuron at a time, so a
    neuron is a modular unit you can add more of.
  </p>
  <p class="read scope">
    <strong>What is and isn't measured.</strong> The drawing is schematic — the bars are
    decorative, not weights. <em>Panels 2 and 3 are real operations</em>
    (<code>mergeAverage</code> / <code>mergeConcat</code>), both run in your browser on the
    loaded parents in Acts 1–2, and the comparison between them is this project's most
    replicated result: concatenation beat averaging in <strong>33 of 33 runs, on all three
    datasets, at every θ</strong>. <em>Panel 1 we did not test</em> — this project contains no
    Transformer. It is a structural claim about the architecture, and it is why the field's
    Transformer merges average weights (arXiv:2203.05482) or add task vectors
    (arXiv:2212.04089) rather than growing capacity.
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
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.4rem;
  }
  .head-right { display: inline-flex; align-items: center; gap: 0.4rem; }
  .title { font-size: 0.83rem; font-weight: 600; color: var(--fg); }
  .toggle {
    border: 1px solid var(--border); background: var(--panel-bg); color: var(--muted);
    border-radius: 999px; padding: 0.15rem 0.55rem; font-size: 0.68rem;
    cursor: pointer; font-family: inherit;
  }

  /* Three panels: a row on a wide screen, a column on a narrow one. Each panel
     is its own SVG in its own coordinate frame, so the direction is pure CSS —
     nothing has to be re-laid-out, and nothing scrolls sideways. */
  .stage { display: flex; align-items: stretch; }
  .panel { flex: 1 1 0; min-width: 0; }
  .panel + .panel { border-left: 1px solid #e2d9bd; }
  svg { width: 100%; height: auto; display: block; }

  .panel-title { font-size: 12.5px; font-weight: 700; fill: #1c1b19; font-family: inherit; }
  .panel-sub { font-size: 9px; fill: #8a8474; font-family: inherit; }
  .glyph-label { font-size: 10px; font-weight: 700; font-family: inherit; }
  .brace { stroke: #6b6459; stroke-width: 1.2; }
  .seam { stroke: #1c1b19; stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.55; }
  .measure { font-size: 10px; font-weight: 700; fill: #4a463f; font-family: inherit; }
  .why { font-size: 9px; fill: #8a8474; font-family: inherit; }
  .verdict { font-size: 9.5px; font-weight: 700; font-family: inherit; }
  .fail-mark { font-size: 11px; font-weight: 700; fill: #a5453b; font-family: inherit; }
  .fail-note { font-size: 8.5px; fill: #a5453b; font-family: inherit; }

  .read { margin: 0; font-size: 0.76rem; line-height: 1.5; color: var(--muted); max-width: 84ch; }
  .read strong { color: var(--fg); }
  .read code { font-size: 0.72rem; background: #f1e9cf; padding: 0 0.2rem; border-radius: 3px; }
  .scope { font-size: 0.71rem; border-left: 2px solid #c9a84c; padding-left: 0.55rem; }

  @media (max-width: 760px) {
    .stage { flex-direction: column; gap: 0.2rem; }
    /* the separator becomes horizontal, and each panel gets the full width —
       so the labels render at full size instead of a third of it */
    .panel + .panel {
      border-left: none;
      border-top: 1px solid #e2d9bd;
      padding-top: 0.35rem;
    }
    /* a stacked panel is much wider than it is tall at full width; cap it so
       three of them do not push the acts far down the page */
    .panel svg { max-height: 46vh; }
  }
</style>
