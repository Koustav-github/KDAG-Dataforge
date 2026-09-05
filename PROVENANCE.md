# Provenance

Source, licence, and AI-assistance disclosure for this submission.

## Code

| Component | Source | Licence | Notes |
|---|---|---|---|
| `model/src/bdh_surgery/bdh.py` | adapted from [pathwaycom/bdh](https://github.com/pathwaycom/bdh) | MIT (confirmed via the GitHub API's `license` field on the repository, not assumed) | Kept byte-identical to upstream, copyright header retained (`Copyright 2025 Pathway Technology, Inc.`). |
| Everything else under `model/src/bdh_surgery/` and all of `web/src/` | original, written for this project | — | `domains.py`, `train.py`, `merge.py`, `damage.py`, `overlap.py`, `ablate.py`, `sweep.py`, `export.py`, and the Svelte/JS front end. |

### One documented deviation from upstream `bdh.py` behaviour

`BDH.forward`'s loss path calls `targets.view(-1)` (see `model/src/bdh_surgery/bdh.py:149`), which raises
on a non-contiguous tensor. A right-shifted slice such as `batch[:, 1:]` is not contiguous. Rather than
modify the vendored file, `model/src/bdh_surgery/train.py` calls `.contiguous()` on both halves at its two
call sites before passing them to the model (`train.py:30` and `train.py:42`). This is a call-site
workaround only — `bdh.py` itself is unmodified from upstream.

## Data

All training and evaluation data is **generated**, not scraped or collected from any external corpus. Two
synthetic languages are built over a shared pivot vocabulary; a parameter θ controls what fraction of
concepts get an identical surface token in both languages (θ = 0: disjoint vocabularies, θ = 1: identical
vocabularies). The layout (token ranges, specials) is derived from `n_concepts` via
`domains.py:layout()`, not hard-coded, so the same generator serves all three datasets below. See
`model/src/bdh_surgery/domains.py` and `model/src/bdh_surgery/datasets.py`.

Three dataset specs share this generator and the identical protocol, varying only one axis each:

| dataset | n_concepts | phrase_len | vocab_size | what it varies |
|---|---|---|---|---|
| `baseline` | 24 | 3 | 96 | — |
| `long_phrase` | 24 | 6 | 96 | phrase length |
| `large_vocab` | 40 | 3 | 128 | vocabulary size |

## Weights

Weights are trained in-repo, from scratch, on the generated synthetic data — no pretrained checkpoints are
used anywhere in the pipeline. Seeds are pinned in `model/src/bdh_surgery/sweep.py`: `SEEDS = (0, 1, 2)`,
swept across `THETAS = (0.0, 0.1, ..., 1.0)` (11 values), giving a 33-row sweep per dataset — 99 rows total
across `baseline`, `long_phrase`, and `large_vocab` — in `web/public/data/sweep.json`
(`schema_version: 2`, nested under `datasets.<id>`).

## Verification

The browser-side (JavaScript) BDH forward pass in `web/src/bdh_forward.js` was checked against the
reference PyTorch implementation and matches to a maximum absolute difference of **3.5e-07**.

## Fonts / graphics

No external fonts or graphic assets are used. The web UI renders exclusively with system fonts and
hand-drawn `<canvas>`/inline-SVG visualisations built from the exported data or, where explicitly labelled
"illustration", as a plain conceptual schematic.

## AI assistance disclosure

This codebase — the Python pipeline under `model/`, the JavaScript/Svelte front end under `web/`, and this
documentation — was implemented with substantial AI assistance (Claude), working from a human-authored plan
and specification (`ps/`, `docs/`). Every task was reviewed by the team before being accepted. This
disclosure is made plainly per the project's evidence-discipline rules.
