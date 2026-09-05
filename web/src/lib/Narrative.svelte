<script>
  // Guides the learner through Acts 1-4 with one short paragraph each, then
  // unlocks every control for free exploration. App.svelte reads
  // `narrativeStep` to dim/lock sections the learner hasn't reached yet;
  // this component only owns the stepper itself.
  import { narrativeStep } from './store.js';
  import HonestyBadge from './HonestyBadge.svelte';

  const ACTS = [
    {
      n: 1,
      title: 'Act 1 — Fuse',
      body: `Two small BDH language models are trained on related but distinct synthetic
        languages. Each shares a θ-controlled fraction of its vocabulary with the other
        (θ = 0: disjoint, θ = 1: identical) — the schematic below shows what that overlap
        means. Below, they're already glued into one merged model: not blended, but
        concatenated end to end along the neuron axis. That operation is available here
        specifically because every BDH parameter already lives on one uniform neuron
        axis — a Transformer, with its fixed per-layer weight shapes, has no equivalent
        move.`,
    },
    {
      n: 2,
      title: 'Act 2 — Concat vs. average',
      body: `Toggle the merge mode and watch the same prompt run through parent A, parent
        B, the merge, and the oracle target. The strongest result in the project, and the
        one that held without a single exception: merging by averaging is worse than
        merging by concatenation — in 33 of 33 individual runs and at 11 of 11 θ values.
        That's the architectural point of this whole artifact.`,
    },
    {
      n: 3,
      title: 'Act 3 — The phase diagram',
      body: `Here's the honest part. Merge damage turns out to be strongly predictable —
        corr(θ, damage) = −0.991 across the 11 per-θ means (3 seeds each), −0.790 across
        all 33 individual runs — falling near-monotonically from +0.769 at θ = 0 to
        +0.218 at θ = 1 (ten of the eleven steps fall; θ = 0.2 → 0.3 rises slightly).
        But that isn't what we set out to measure. Our representational-overlap metric M
        (Hungarian-matched neuron activation correlation) stays almost flat: 0.704 to
        0.728 across the 11 per-θ means, 0.680 to 0.742 across all 33 runs, while damage
        swings by 0.551 over those means and 0.849 over the individual runs. And
        corr(M, damage) is −0.608 over the 11 per-θ means but +0.174 over all 33
        individual runs — the sign flips when you stop averaging. That's noise riding on
        a metric that barely moves, not a relationship. We re-probed each parent on its
        own target direction to rule out a broken probe; M went 0.736 → 0.757, still
        flat. The probe isn't the problem — the metric doesn't capture what actually
        drives mergeability. Our reading (interpretation, not fact): both parents are
        fine-tuned clones of one shared base at a modest learning rate, so most neurons
        stay near the common ancestor and match well under every condition, while the
        divergence that actually governs merge damage concentrates in the output pathway
        — exactly what a mean-over-all-neurons statistic washes out. That's a genuine
        partial answer to a question the BDH paper (§7.1) poses but never tests, not a
        refutation of it. Two caveats on the other things you'll see here. The self-merge
        floor fuses a model with a bit-identical duplicate of itself (not an
        independently seeded retrain) and lands at −0.0008 to +0.0035 damage across the
        33 conditions; the pipeline is deterministic, so that spread is variation across
        θ and seed, not run-to-run noise — concatenating a model with its own duplicate
        is near-lossless, which is narrower than it sounds but still checks the pipeline
        end to end. And damage is larger on one side than the other at every θ (at θ = 1:
        0.344 vs. 0.091). We used to call that the paper's into-versus-out-of-pivot
        asymmetry; it isn't. Both of our eval sets run pivot → target, so there is no
        into-pivot direction here at all — what we see is a difference between the two
        target languages, and it is thin: the per-seed ratios at θ = 1 are 11.2× / 2.45×
        / 1.19×, so seed 0 carries almost all of it.`,
    },
    {
      n: 4,
      title: 'Act 4 — Surgery',
      body: `Find the "collision set" — neurons active on both parents' directions at
        once — and ablate them. A size-matched random ablation runs alongside, because
        ablating any neuron perturbs attention for every other neuron by roughly 2% of
        typical activation scale (x_sparse feeds attention scores that sum over the
        whole neuron axis); that's intrinsic to the mechanism, not a bug, and it's why
        the reportable number is the gap between collision- and random-ablation loss,
        never the raw recovery on its own. Read that gap honestly, because it does not
        go our way. We expected the collision set to be where the damage lives, so that
        cutting it would recover function. It isn't. Collision ablation never drops the
        loss below baseline, and by k = 100 it is clearly worse than cutting the same
        number of neurons at random (at θ = 0.5, k = 100: baseline 2.019, collision
        2.089, random 1.976). At k = 40 the gap is inside the random control's own
        spread. So the second half of our original hypothesis — that merge damage
        localizes to identifiable neurons — is unsupported by this experiment. That
        rules out this particular collision score, not the idea; it pools over the whole
        neuron axis, exactly like M does.`,
    },
  ];

  $: activeAct = ACTS.find((a) => a.n === $narrativeStep) ?? null;
  $: isFree = $narrativeStep === 'free';

  function next() {
    if (typeof $narrativeStep !== 'number') return;
    narrativeStep.set($narrativeStep >= 4 ? 'free' : $narrativeStep + 1);
  }
  function back() {
    if (typeof $narrativeStep !== 'number' || $narrativeStep <= 1) return;
    narrativeStep.set($narrativeStep - 1);
  }
  function skip() {
    narrativeStep.set('free');
  }
  function restart() {
    narrativeStep.set(1);
  }
</script>

<div class="narrative" class:free={isFree}>
  {#if !isFree && activeAct}
    <div class="guide-card">
      <div class="guide-head">
        <span class="step-dots" aria-hidden="true">
          {#each ACTS as a}
            <span class="dot" class:done={a.n < $narrativeStep} class:current={a.n === $narrativeStep}></span>
          {/each}
        </span>
        <h2>{activeAct.title}</h2>
      </div>

      <p class="guide-body">{activeAct.body}</p>

      {#if activeAct.n === 1}
        <div class="illustration-block">
          <div class="illustration-head">
            <span class="illustration-label">what θ means</span>
            <HonestyBadge kind="illustration" />
          </div>
          <svg viewBox="0 0 220 90" role="img" aria-label="Schematic: two circles overlapping to represent shared vocabulary at a given θ">
            <circle cx="85" cy="45" r="38" class="circ circ-a" />
            <circle cx="135" cy="45" r="38" class="circ circ-b" />
            <text x="55" y="49" class="circ-label">A only</text>
            <text x="110" y="49" class="circ-label overlap-label">shared (θ)</text>
            <text x="165" y="49" class="circ-label">B only</text>
          </svg>
          <p class="illustration-caption">
            Schematic only — not measured data. θ = 0 pulls the circles apart entirely; θ = 1
            makes them coincide.
          </p>
        </div>
      {/if}

      <div class="guide-controls">
        <button class="ghost" on:click={back} disabled={$narrativeStep <= 1}>back</button>
        <button class="link" on:click={skip}>skip to free exploration</button>
        <button class="primary" on:click={next}>
          {$narrativeStep >= 4 ? 'start exploring' : 'next'}
        </button>
      </div>
    </div>
  {:else}
    <div class="free-banner">
      <span>You've seen the whole arc — every control below is unlocked. Pick any θ, merge mode, or ablation set.</span>
      <button class="link" on:click={restart}>replay the narrative</button>
    </div>
  {/if}
</div>

<style>
  .narrative {
    display: flex;
    flex-direction: column;
  }
  .guide-card {
    border: 1px solid var(--accent);
    background: var(--accent-weak);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }
  .guide-head {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    flex-wrap: wrap;
  }
  .guide-head h2 {
    margin: 0;
    font-size: 1rem;
  }
  .step-dots {
    display: flex;
    gap: 0.3rem;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--border);
  }
  .dot.done {
    background: var(--accent);
    opacity: 0.5;
  }
  .dot.current {
    background: var(--accent);
  }
  .guide-body {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--fg);
    max-width: 74ch;
  }
  .illustration-block {
    border: 1px dashed #c9a84c;
    background: #fbf6e6;
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    max-width: 320px;
  }
  .illustration-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .illustration-label {
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
  }
  .illustration-block svg {
    width: 100%;
    height: auto;
  }
  .circ {
    fill-opacity: 0.35;
    stroke-width: 1.5;
  }
  .circ-a {
    fill: #3b6ea5;
    stroke: #3b6ea5;
  }
  .circ-b {
    fill: #a5453b;
    stroke: #a5453b;
  }
  .circ-label {
    font-size: 8px;
    fill: var(--fg);
    text-anchor: middle;
  }
  .overlap-label {
    font-weight: 600;
  }
  .illustration-caption {
    margin: 0;
    font-size: 0.68rem;
    color: var(--muted);
  }
  .guide-controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  button {
    border: 1px solid var(--border);
    background: var(--panel-bg);
    color: var(--fg);
    border-radius: 6px;
    padding: 0.4rem 0.85rem;
    font-size: 0.8rem;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  button.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #ffffff;
    margin-left: auto;
  }
  button.ghost {
    background: transparent;
  }
  button.link {
    border-color: transparent;
    background: transparent;
    color: var(--muted);
    text-decoration: underline;
  }
  .free-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.8rem;
    color: var(--muted);
    background: var(--panel-bg);
  }

  @media (max-width: 640px) {
    .guide-controls {
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    /* the illustration cap is narrower than most phone viewports already;
       let it use the full width rather than sitting in a 320px column */
    .illustration-block { max-width: none; }
  }

  @media (pointer: coarse) {
    button { padding: 0.55rem 0.95rem; }
  }
</style>
