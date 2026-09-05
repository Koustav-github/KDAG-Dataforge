<script>
  // Four columns generated from one shared prompt: what parent A predicts,
  // what parent B predicts, what the merged model predicts, and the oracle
  // (ground-truth) continuation from probes.json. Each of parentA/parentB/
  // merged is { prompt, continuation, badge }; oracle is { prompt, target }.
  export let parentA = null;
  export let parentB = null;
  export let merged = null;
  export let oracle = null;

  const columns = [
    { key: 'a', title: 'Parent A', badge: 'live' },
    { key: 'b', title: 'Parent B', badge: 'live' },
    { key: 'merged', title: 'Merged', badge: 'live' },
    { key: 'oracle', title: 'Oracle target', badge: 'truth' },
  ];

  $: data = { a: parentA, b: parentB, merged, oracle };
</script>

<div class="panels">
  {#each columns as col (col.key)}
    <div class="panel">
      <div class="panel-head">
        <span class="title">{col.title}</span>
        <span class="badge badge-{col.badge}">{col.badge}</span>
      </div>
      {#if data[col.key]}
        <div class="row prompt">
          {#each data[col.key].prompt as tok}<span class="tok">{tok}</span>{/each}
        </div>
        <div class="row continuation">
          {#each (data[col.key].continuation ?? data[col.key].target) as tok}<span class="tok">{tok}</span>{/each}
        </div>
      {:else}
        <div class="row muted">loading…</div>
      {/if}
    </div>
  {/each}
</div>

<style>
  .panels {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
  }
  .panel {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.7rem;
    background: var(--panel-bg);
    min-width: 0;
  }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  .title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--fg);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem;
    min-height: 1.4rem;
  }
  .row.prompt {
    opacity: 0.55;
    margin-bottom: 0.3rem;
  }
  .row.muted {
    opacity: 0.5;
    font-size: 0.8rem;
  }
  .tok {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 0.75rem;
    background: var(--tok-bg);
    border-radius: 3px;
    padding: 0.05rem 0.3rem;
    white-space: nowrap;
  }
  .row.continuation .tok {
    background: var(--accent-weak);
    font-weight: 600;
  }
  .badge {
    font-size: 0.62rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    border: 1px solid transparent;
  }
  .badge-live {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--accent-weak);
  }
  .badge-truth {
    color: var(--muted);
    border-color: var(--border);
    background: transparent;
  }
  @media (max-width: 760px) {
    .panels {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
