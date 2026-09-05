<script>
  // A small honesty label applied throughout the artifact so the learner
  // always knows what kind of thing they're looking at:
  //   live         - computed in the browser right now, from the loaded
  //                  weights (forward passes, merges, ablations).
  //   precomputed  - produced offline by the Python sweep and shipped as
  //                  static data (sweep.json); the browser only reads it.
  //   illustration - a schematic drawn to explain a concept, not a
  //                  rendering of measured data at all.
  export let kind = 'live'; // 'live' | 'precomputed' | 'illustration'
  export let label = null; // optional override text

  const TEXT = {
    live: 'live',
    precomputed: 'precomputed',
    illustration: 'illustration',
  };

  const TITLE = {
    live: 'Computed in your browser right now, from the loaded weights.',
    precomputed: 'Produced offline by the Python sweep (sweep.json) and shipped as static data.',
    illustration: 'A schematic to explain a concept — not a rendering of measured data.',
  };

  $: text = label ?? TEXT[kind] ?? kind;
  $: title = TITLE[kind] ?? '';
</script>

<span class="honesty-badge honesty-{kind}" {title}>{text}</span>

<style>
  .honesty-badge {
    display: inline-block;
    font-size: 0.62rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    border: 1px solid transparent;
    line-height: 1.4;
    white-space: nowrap;
    cursor: default;
  }
  .honesty-live {
    color: var(--accent, #1f6f5c);
    border-color: var(--accent, #1f6f5c);
    background: var(--accent-weak, #e1efe9);
  }
  .honesty-precomputed {
    color: var(--muted, #726f68);
    border-color: var(--border, #ddd9d1);
    background: transparent;
  }
  .honesty-illustration {
    color: #8a6d1f;
    border-color: #c9a84c;
    background: #f7efd8;
  }
</style>
