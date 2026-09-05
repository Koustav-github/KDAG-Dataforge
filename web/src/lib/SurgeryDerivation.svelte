<script>
  // Act 4's working, in the same shape as Act 3's: the formula written out,
  // then evaluated on this model's actual numbers, one step at a time.
  //
  // Nothing here is illustrative. The neuron shown in step 1 is the real
  // top-scoring one, its a/b activations come from live forward passes on the
  // probe sets, and the losses are the same values the chart above plots.
  // GSAP controls only when each figure appears, never what it is.
  import { onDestroy, tick } from 'svelte';
  import gsap from 'gsap';

  export let breakdown = null;   // { a, b, joint, imbalance, scores } | null
  export let collisionSet = [];
  export let k = 40;
  export let baseline = null;
  export let collisionLoss = null;
  export let randomMean = null;
  export let randomSD = null;
  export let nDraws = 0;
  export let busy = false;

  const A_COLOR = '#0a8f72';
  const B_COLOR = '#c25e12';
  const STEP = 900;

  let step = 0;
  const LAST = 5;
  let tl = null;
  let anim = { a: 0, b: 0, joint: 0, imb: 0, score: 0, thresh: 0, gap: 0, sig: 0 };

  const reduceMotion =
    typeof window !== 'undefined' && window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // The highest-scoring neuron: the one the collision metric is most confident
  // about, and therefore the fairest one to show the formula working on.
  $: top = (() => {
    if (!breakdown || !collisionSet.length) return null;
    const i = collisionSet[0];
    return {
      idx: i,
      a: breakdown.a[i],
      b: breakdown.b[i],
      joint: breakdown.joint[i],
      imbalance: breakdown.imbalance[i],
      score: breakdown.scores[i],
    };
  })();

  // The k-th score — everything at or above it gets ablated.
  $: threshold = breakdown && collisionSet.length
    ? breakdown.scores[collisionSet[collisionSet.length - 1]]
    : null;

  $: gap = collisionLoss !== null && randomMean !== null ? randomMean - collisionLoss : null;
  $: sigmas = gap !== null && randomSD ? Math.abs(gap) / randomSD : null;

  $: ready = top !== null && baseline !== null && collisionLoss !== null && randomMean !== null;
  $: if (ready && !busy) restart(top, k, collisionLoss, randomMean);

  async function restart() {
    if (tl) tl.kill();
    step = 0;
    anim = { a: 0, b: 0, joint: 0, imb: 0, score: 0, thresh: 0, gap: 0, sig: 0 };
    await tick();
    if (!ready) return;

    const target = {
      a: top.a, b: top.b, joint: top.joint, imb: top.imbalance, score: top.score,
      thresh: threshold ?? 0, gap: gap ?? 0, sig: sigmas ?? 0,
    };

    if (reduceMotion) { step = LAST; anim = { ...target }; return; }

    const D = STEP / 1000;
    tl = gsap.timeline({ onComplete: () => { anim = { ...target }; } });
    const to = (props, onStart, at) =>
      tl.to(anim, { ...props, duration: D * 0.8, ease: 'power1.out',
                    onStart, onUpdate: () => (anim = anim) }, at);

    to({ a: target.a, b: target.b }, () => (step = 1));
    to({ joint: target.joint, imb: target.imb }, () => (step = 2), `+=${D * 0.2}`);
    to({ score: target.score }, () => (step = 3), `+=${D * 0.2}`);
    to({ thresh: target.thresh }, () => (step = 4), `+=${D * 0.2}`);
    to({ gap: target.gap, sig: target.sig }, () => (step = LAST), `+=${D * 0.2}`);
  }

  onDestroy(() => { if (tl) tl.kill(); });

  const f = (v, p = 4) => (v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(p));
</script>

<div class="deriv">
  <div class="head">
    <span class="title">How a neuron gets picked, and how the verdict is reached</span>
    <span class="mode {busy ? 'pre' : 'live'}">{busy ? 'computing…' : 'live'}</span>
  </div>

  {#if !ready}
    <p class="muted">waiting on both arms…</p>
  {:else}
    <ol class="steps">
      <!-- 1 -->
      <li class:on={step >= 1}>
        <span class="n">1</span>
        <div class="body">
          <p class="lead">
            For every neuron, measure how hard it fires on each parent's direction.
            Neuron <strong>#{top.idx}</strong> is the current top scorer.
          </p>
          <p class="calc">
            <code>
              a = mean activation on A's probes = <strong style="color:{A_COLOR}">{f(anim.a)}</strong>
              &nbsp;·&nbsp;
              b = on B's probes = <strong style="color:{B_COLOR}">{f(anim.b)}</strong>
            </code>
          </p>
        </div>
      </li>

      <!-- 2 -->
      <li class:on={step >= 2}>
        <span class="n">2</span>
        <div class="body">
          <p class="lead">
            Two things matter: it must fire on <em>both</em>, and fire on them <em>equally</em>.
            A neuron loud on one direction only is doing its parent's job, not colliding.
          </p>
          <p class="calc">
            <code>joint = min(a, b) = <strong>{f(anim.joint)}</strong></code>
          </p>
          <p class="calc">
            <code>imbalance = |a − b| / (a + b + 1e-8) = <strong>{f(anim.imb)}</strong></code>
            <span class="hint">0 = perfectly balanced, 1 = entirely one-sided</span>
          </p>
        </div>
      </li>

      <!-- 3 -->
      <li class:on={step >= 3}>
        <span class="n">3</span>
        <div class="body">
          <p class="lead">The score rewards firing on both and punishes lopsidedness.</p>
          <p class="calc big">
            <code>
              score = joint × (1 − imbalance) = {f(anim.joint)} × {f(1 - anim.imb)} =
              <strong>{f(anim.score)}</strong>
            </code>
          </p>
        </div>
      </li>

      <!-- 4 -->
      <li class:on={step >= 4}>
        <span class="n">4</span>
        <div class="body">
          <p class="lead">
            Score all {breakdown ? breakdown.scores.length : '—'} neurons, sort, keep the top {k}.
            Those are the ones zeroed — the ringed-and-slashed cells in the map above.
          </p>
          <p class="calc">
            <code>cut-off = {k}-th highest score = <strong>{f(anim.thresh)}</strong></code>
            <span class="hint">everything at or above this is ablated</span>
          </p>
        </div>
      </li>

      <!-- 5 -->
      <li class:on={step >= LAST} class="final">
        <span class="n">5</span>
        <div class="body">
          <p class="lead">
            Now the only comparison that counts: targeted removal against removing the same
            number at random.
          </p>
          <p class="calc">
            <code>
              baseline {f(baseline, 3)} &nbsp;·&nbsp;
              collision <strong style="color:{A_COLOR}">{f(collisionLoss, 3)}</strong> &nbsp;·&nbsp;
              random ×{nDraws} <strong style="color:{B_COLOR}">{f(randomMean, 3)} ± {f(randomSD, 3)}</strong>
            </code>
          </p>
          <p class="calc big">
            <code>
              gap = random − collision = <strong>{anim.gap >= 0 ? '+' : '−'}{f(Math.abs(anim.gap), 3)}</strong>
              &nbsp;=&nbsp; <strong>{f(anim.sig, 1)}σ</strong> of the control's own spread
            </code>
          </p>
          <p class="verdict" class:bad={gap !== null && gap <= 0}>
            {#if gap === null}
              —
            {:else if gap <= 0}
              Negative: targeting did <strong>worse</strong> than chance. The metric identifies
              neurons, but removing them does not repair the merge — so the damage is not
              localized in the way the claim supposed.
            {:else if sigmas !== null && sigmas < 1}
              Positive but under 1σ — smaller than the random draws vary among themselves, so
              not distinguishable from luck.
            {:else}
              Positive and beyond the control's spread: targeting genuinely beat chance here.
            {/if}
          </p>
        </div>
      </li>
    </ol>

    <p class="foot">
      Every number above is computed in your browser from the weights on screen, using the same
      formulas as <code>model/src/bdh_surgery/ablate.py</code>. Change k and the whole derivation
      re-runs.
    </p>
  {/if}
</div>

<style>
  .deriv { display: flex; flex-direction: column; gap: 0.55rem; }
  .head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
  .title { font-size: 0.85rem; font-weight: 600; }
  .mode { font-size: 0.66rem; padding: 0.05rem 0.4rem; border-radius: 999px; }
  .mode.live { background: #e1efe9; color: #0a5f4d; }
  .mode.pre { background: #f3e7d8; color: #8a4a10; }
  .muted { color: var(--muted); font-size: 0.85rem; }
  .steps { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.5rem; }
  .steps li {
    display: flex; gap: 0.6rem; align-items: flex-start;
    opacity: 0.25; transform: translateY(4px);
    transition: opacity 300ms ease, transform 300ms ease;
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
  .lead { margin: 0; font-size: 0.79rem; max-width: 76ch; }
  .calc { margin: 0; font-size: 0.78rem; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: baseline; }
  .calc code { font-family: ui-monospace, Consolas, monospace; font-size: 0.76rem; }
  .calc.big code { font-size: 0.88rem; }
  .hint { font-size: 0.68rem; color: var(--muted); }
  .verdict {
    margin: 0.3rem 0 0; font-size: 0.78rem; line-height: 1.5; max-width: 74ch;
    border-left: 3px solid var(--accent); padding-left: 0.55rem;
  }
  .verdict.bad { border-left-color: #c25e12; }
  .foot { margin: 0; font-size: 0.7rem; color: var(--muted); line-height: 1.45; max-width: 76ch; }
  .foot code { font-family: ui-monospace, Consolas, monospace; font-size: 0.68rem; }
  @media (prefers-reduced-motion: reduce) { .steps li { transition: none; transform: none; } }
</style>
