<script>
  // Renders the merged neuron axis to a <canvas>: under concat, neurons
  // 0..splitAt are parent A's (colour A) and splitAt..n are parent B's
  // (colour B); under average every neuron is the same blended colour,
  // signalled by splitAt === null. `highlight` marks ablated indices
  // (unused until Task 14, but already wired through).
  export let n = 0;
  export let splitAt = null;
  export let highlight = [];

  const COLOR_A = [59, 110, 165]; // #3b6ea5
  const COLOR_B = [165, 69, 59]; // #a5453b
  const blend = COLOR_A.map((c, i) => Math.round((c + COLOR_B[i]) / 2));

  let canvasEl;

  function rgb(c) {
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }

  function draw() {
    if (!canvasEl || !n) return;
    const ctx = canvasEl.getContext('2d');
    canvasEl.width = n;
    canvasEl.height = 40;
    ctx.clearRect(0, 0, n, 40);

    if (splitAt === null || splitAt === undefined) {
      ctx.fillStyle = rgb(blend);
      ctx.fillRect(0, 0, n, 40);
    } else {
      ctx.fillStyle = rgb(COLOR_A);
      ctx.fillRect(0, 0, splitAt, 40);
      ctx.fillStyle = rgb(COLOR_B);
      ctx.fillRect(splitAt, 0, n - splitAt, 40);
    }

    if (highlight && highlight.length) {
      ctx.fillStyle = '#111111';
      for (const idx of highlight) {
        if (idx >= 0 && idx < n) ctx.fillRect(idx, 0, 1, 8);
      }
    }
  }

  $: draw(n, splitAt, highlight, canvasEl);
</script>

<div class="strip">
  <!-- svelte-ignore a11y_no_interactive_element_to_noninteractive_role -->
  <canvas
    bind:this={canvasEl}
    width={n || 1}
    height="40"
    role="img"
    aria-label={splitAt === null || splitAt === undefined
      ? `Merged neuron axis of ${n} neurons, averaged into a single blended color`
      : `Merged neuron axis of ${n} neurons: parent A occupies neurons 0 through ${splitAt}, parent B occupies neurons ${splitAt} through ${n}`}
  ></canvas>
  <div class="legend">
    {#if splitAt === null || splitAt === undefined}
      <span class="swatch blend"></span> averaged, n={n}
    {:else}
      <span class="swatch a"></span> parent A, 0..{splitAt}
      <span class="swatch b"></span> parent B, {splitAt}..{n}
    {/if}
  </div>
</div>

<style>
  .strip {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  canvas {
    width: 100%;
    height: 40px;
    display: block;
    border-radius: 4px;
    image-rendering: pixelated;
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--muted);
  }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-left: 0.6rem;
  }
  .swatch:first-child {
    margin-left: 0;
  }
  .swatch.a {
    background: rgb(59, 110, 165);
  }
  .swatch.b {
    background: rgb(165, 69, 59);
  }
  .swatch.blend {
    background: rgb(112, 90, 112);
  }
</style>
