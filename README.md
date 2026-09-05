# Model Surgery: fusing two BDH language models

Team Invariance, IIT Kharagpur. Submission for the Pathway track: an interactive
explainer connecting **model merging / composability in BDH** (arXiv:2509.26507 §7.1)
to the concept of parametric memory in LLMs. See `ps/` (the two problem-statement PDFs)
and `docs/` for the original design rationale.

## The claim

> Merge damage in a pair of BDH models is strongly predictable — but not from the thing
> we set out to measure. It is predictable from **θ**, how much the two models' training
> vocabularies overlap: `corr(θ, damage) = -0.991` across the 11 per-θ means (3 seeds
> each), and `-0.790` across all 33 individual runs. Damage falls near-monotonically
> from **+0.769** at θ = 0 to **+0.218** at θ = 1 (ten of the eleven steps fall;
> θ = 0.2 → 0.3 rises slightly, 0.638 → 0.647). Our measured representational overlap
> **M** — Hungarian-matched neuron activation correlation between the two parents —
> stays almost flat over that whole range: **0.704 to 0.728** across the 11 per-θ means
> (0.680 to 0.742 across all 33 runs), against a damage spread of 0.551 over the per-θ
> means (0.849 across all 33 runs). And `corr(M, damage)` is **-0.608 over the 11 per-θ
> means but +0.174 over all 33 individual runs — the sign flips when you stop
> averaging.** That is what noise on a metric that barely moves looks like; it is not a
> relationship. We ruled out a broken probe by re-probing each parent on its own target
> direction instead of the shared pivot; it didn't help (M went 0.736 → 0.757, still
> flat — see `model/scripts/probe_diagnostic.py`). **The probe was not the problem — the
> metric does not capture what drives mergeability.**

This is not the claim the project set out to test, and **neither half of the original
hypothesis survived it.** We set out to show that mergeability is predictable *from
measured neuron overlap*, and that collision damage *localizes to identifiable neurons*.
The first half is falsified by our own data (above). The second half we had never
actually measured until this review; when we did measure it (`model/scripts/measure_locality.py`,
table below), ablating the collision set never recovered function, and at k = 100 it was
significantly *worse* than ablating the same number of random neurons. **Both halves are
unsupported by this experiment.**

That is a real negative result about an open question, not a broken project. The
averaging control and the θ relationship are genuine positive findings and they stand on
their own. We report all of it as a partial answer to a question arXiv:2509.26507 §7.1
poses but never tests ("when the model latent space promotes concept disentangling then
it is feasible to directly compose concepts"), not as a refutation of the paper. Our
interpretation (stated as interpretation, not fact):
both parents in this project are fine-tuned clones of one shared base at a modest
learning rate (see `model/src/bdh_surgery/train.py`), so the bulk of the neuron
population stays near the common ancestor and matches well under every condition. The
divergence that actually drives merge damage concentrates in the output pathway, which a
mean-over-all-neurons statistic like M washes out.

### The locality half, measured — and retracted

Running the shipped `web/src/collision.js` + `web/src/bdh_forward.js` against the real
exported weights (`model/scripts/measure_locality.py`, teacher-forced mean cross-entropy
over the first four rows of each eval set — exactly what Act 4 computes in the browser):

| θ | k | baseline | collision-ablated | random (seed 7) | headline (random − collision) |
|---|---|---|---|---|---|
| 0.0 | 40 | 1.5881 | 1.5871 | 1.5886 | **+0.0014** |
| 0.5 | 40 | 1.3534 | 1.3569 | 1.3447 | **−0.0122** (the artifact's shipped default) |
| 1.0 | 40 | 1.1169 | 1.1154 | 1.1210 | **+0.0057** |
| 0.0 | 100 | 1.5881 | 1.6488 | 1.5951 | **−0.0536** |
| 0.5 | 100 | 1.3534 | 1.4347 | 1.4017 | **−0.0330** |
| 1.0 | 100 | 1.1169 | 1.2021 | 1.1881 | **−0.0139** |

Collision ablation never recovers function — the collision-ablated loss is at or above
baseline in four of six cells, and both cells below it (θ = 0 and θ = 1 at k = 40) are
below by 0.0010 and 0.0015, well inside the 0.011–0.023 spread of the random control at
that k. At k = 100 the collision set is *worse* than random at every θ, by 0.4–1.9× the
random draws' standard
deviation. Damage does not localize to the neurons this collision metric identifies.

We report this as a negative result on an open question. What it rules out is our
particular collision score (`joint × (1 − imbalance)` on mean sparse activation); it does
not rule out that merge damage localizes to *some* identifiable set — measuring overlap
and collision in the output pathway specifically, rather than pooling over the whole
neuron axis, is the obvious next thing to try, and is the same suspicion M's flatness
raises.

### What *did* replicate

- **The averaging control holds without a single exception**: merging by averaging is
  worse than merging by concatenation in **33 of 33 individual runs and at 11 of 11 θ
  values**. This is the strongest result in the project, and it is the architectural
  point of the whole artifact — concatenation is available to BDH because every
  parameter lives on one uniform neuron axis; no Transformer has an equivalent
  operation.
- **Self-merge floor**: fusing a model with a `copy.deepcopy` duplicate of itself (see
  `sweep.py`: it is a bit-identical copy, not an independently seeded retrain) yields
  damage of −0.00077 ≤ D ≤ +0.00347 across all 33 runs, mean +0.00135, with 3 of the 33
  landing marginally *below* zero. Note what this does and does not establish: with
  `dropout = 0.0` and `.eval()` the pipeline is fully deterministic, so this spread is
  **not** run-to-run noise — it is the spread across the 11 θ and 3 seed conditions,
  each of which produces a different parent to duplicate. What the control shows is
  narrower than "the merge is lossless to within noise": concatenating a model with an
  exact duplicate of itself is near-lossless, which is a real end-to-end check on the
  merge, ablation-free forward pass, and damage measurement, but it does not probe
  seed-to-seed representational variation at all.
- **A directional difference in damage, which is *not* the paper's asymmetry**: damage
  is larger on one side than the other at every θ (at θ = 1, pooled over seeds:
  d_b = +0.344 vs. d_a = +0.091, 3.78×). We previously described this as reproducing the
  paper's finding. **That was a category error and we retract it.** The paper's
  asymmetry is translation *into* the shared pivot language versus *out of* it. Both of
  our eval sets run pivot → target: `domains.py:make_corpus` emits `BOS p… SEP t… EOS`
  for direction A and direction B alike, so there is no into-pivot direction anywhere in
  this pipeline to be asymmetric with. What we observe is a difference between the two
  *target languages* under merge, which is a different thing. It is also thin evidence:
  the per-seed d_b/d_a ratios at θ = 1 are **11.2× / 2.45× / 1.19×** — seed 0 dominates
  the pooled number almost entirely.

### The open question this leaves

**Both negative results point the same way, and that is the most important open
result.** We cannot yet say whether representational overlap is genuinely non-predictive
of mergeability, or whether our particular metric (mean Hungarian-matched activation
correlation across the whole neuron axis) is simply mis-specified for this architecture.
The same applies to the collision score, which is built on the same
mean-over-all-neurons activation profile: we cannot say whether merge damage is
genuinely non-local, or whether we looked for it in the wrong place. Both metrics pool
over the entire neuron axis; the divergence we suspect actually matters is in the output
pathway. Resolving either would mean measuring there specifically, rather than pooling
over every neuron — and that is the experiment we would run next.

### Does any of this survive a different dataset?

We ran the identical protocol — same architecture, same clone-then-diverge training,
same 11 θ × 3 seeds sweep — on two more synthetic datasets, and put a selector in the
artifact so you can switch between all three yourself:

| dataset | what varies | corr(θ,D)<br>per-θ / all-33 | corr(M,D)<br>per-θ / all-33 | M spread |
|---|---|---|---|---|
| baseline | — | −0.991 / −0.790 | −0.608 / **+0.174** | 0.024 |
| long_phrase | 6-token phrases (was 3) | −0.991 / −0.924 | −0.709 / −0.204 | 0.041 |
| large_vocab | 40 concepts (was 24) | −0.990 / −0.972 | **−0.935** / −0.342 | 0.023 |

Two things held with no exceptions across all three, 33/33 runs each: **averaging is
worse than concatenation at every single θ**, and the self-merge floor stays at
essentially zero. Those are now the best-replicated results in the project.

**M's flatness did not replicate.** At 24 concepts (baseline), M sits at a nearly
constant ~0.72 regardless of θ, and its correlation with damage flips sign depending on
how you aggregate — the finding we built the "M does not predict mergeability" claim on.
At 40 concepts (large_vocab), M *rises* — 0.701 at θ = 0 to 0.723 at θ = 1 — on a spread
barely larger than baseline's (0.023 vs 0.024), and correlates with damage more strongly
than θ does at the per-θ level (−0.935). Same tiny spread, opposite behaviour.

That revises rather than overturns the earlier claim: **M's flatness is not a property
of this architecture, it is sensitive to vocabulary size.** Whatever makes the
neuron-overlap metric go flat at 24 concepts stops happening at 40. We do not know why —
that is now the sharper version of the open question above, and it is a genuinely
different, more interesting finding than "the metric is broken": it means the
overlap-mergeability relationship is conditional on task parameters a single toy
dataset cannot reveal, which is itself evidence against dismissing the metric outright.

## Intended learner and prerequisites

Aimed at someone who already knows roughly what a language model is and has heard of
model merging (e.g. weight averaging, task vectors) but hasn't seen BDH (Baby Dragon
Hatchling, arXiv:2509.26507) or thought about *why* an architecture's parameter layout
determines which merge operations are even well-defined. No BDH-specific background is
assumed — the artifact explains the neuron-axis idea from scratch — but basic familiarity
with next-token prediction and cross-entropy loss is assumed.

## Learning objectives

By the end of the guided narrative, the learner should be able to:

1. Explain why BDH's uniform neuron axis makes concatenation a well-defined merge
   operation, and why that has no equivalent in a standard Transformer.
2. State the (revised) central finding: merge damage is predictable from vocabulary
   overlap θ, not from the representational-overlap metric measured here — and explain
   why that is a genuine open question rather than a settled negative result.
3. Read a collision-ablation vs. random-ablation comparison correctly: understand why
   the *difference* between the two arms is the reportable number, not the raw loss
   recovery from ablating the collision set alone — and read this artifact's own
   difference correctly, which is negative.
4. Distinguish, for any number or chart in the artifact, whether it was computed live in
   the browser, precomputed offline by the Python sweep, generated synthetic data, or a
   non-data illustration — using the honesty badges described below.

## Artifact architecture

```
model/                          Python (uv project)
  src/bdh_surgery/
    bdh.py                      BDH-GPU model, adapted from pathwaycom/bdh (unmodified)
    domains.py                  synthetic two-language data generator, θ-controlled overlap
    train.py                    base pretrain + per-direction fine-tune (clone protocol)
    merge.py                    the concat/average merge rule (arXiv:2509.26507 §7.1)
    overlap.py                  M: Hungarian-matched neuron activation correlation
    ablate.py                   collision-neuron scoring (fires on both parents' directions)
    damage.py                   D: merge damage relative to the parent it replaces
    recurrent.py                the recurrent (Eq. 8) forward pass, ported to JS for the browser
    sweep.py                    the θ x seed grid that produces sweep.json (33 rows)
    export.py                   writes web/public/data/ (manifest, weights, probes, sweep)
  scripts/
    measure_locality.py         the locality check: drives the shipped JS via node
    probe_diagnostic.py         M under the shared-pivot probe vs. own-direction probes
  tests/                        44 tests covering every module above

web/                             Svelte + Vite static site
  src/
    bdh_forward.js               JS port of recurrent.py's forward pass (verified to 3.5e-07)
    merge.js                     client-side concat/average merge (mirrors merge.py)
    collision.js                 client-side collision scoring (mirrors ablate.py)
    App.svelte                   page shell: loads data, wires the four acts together
    lib/
      Narrative.svelte           guided walkthrough, Acts 1-4, then unlocks free exploration
      HonestyBadge.svelte        live / precomputed / illustration labelling
      NeuronStrip.svelte         Act 1: the merged neuron axis, parent-A/parent-B split
      OutputPanels.svelte        Act 2: four-way generation comparison (A, B, merged, oracle)
      PhaseDiagram.svelte        Act 3: M vs. D across the θ sweep, hand-rolled inline SVG
      Surgery.svelte             Act 4: collision-set ablation vs. random-ablation control
      store.js, tokens.js, generate.js   shared reactive state, vocabulary labelling, decoding
  public/data/                   exported artifacts: manifest.json, probes.json, sweep.json,
                                  and every swept θ's parent weights as int8 .bin (dequantized on load)
```

## Role of every major component

| File | Role |
|---|---|
| `model/src/bdh_surgery/bdh.py` | The BDH-GPU model definition (config, attention, forward pass, loss). Vendored from upstream and kept byte-identical — see `PROVENANCE.md` for the one call-site workaround this required in `train.py`. |
| `model/src/bdh_surgery/domains.py` | Generates two synthetic "languages" sharing a θ fraction of a pivot vocabulary. This is the only source of training/eval data in the project — nothing is scraped. Token ranges are computed from `n_concepts` (`layout()`), not hard-coded, which is what lets `large_vocab` use 40 concepts without its target pool colliding with the special tokens — at n_concepts=24 this reproduces the original fixed layout exactly (tested). |
| `model/src/bdh_surgery/datasets.py` | The three dataset specs (`baseline`, `long_phrase`, `large_vocab`): what each varies, and nothing else — same architecture, same protocol, same θ/seed grid throughout, so a difference between them is attributable to the data alone. |
| `model/src/bdh_surgery/train.py` | Trains one shared base per seed, then clones and fine-tunes it in each of the two domain directions — mirroring the paper's §7.1 protocol. |
| `model/src/bdh_surgery/merge.py` | Implements the paper's merge rule: concatenate every tensor carrying the neuron dimension n; average everything else (LayerNorm is parameter-free and needs no handling). |
| `model/src/bdh_surgery/overlap.py` | Computes M: for each pair of parents, Hungarian-matches neurons by activation profile and reports the matched correlation, thresholded at τ ∈ {0.5, 0.7, 0.9} plus a thresholdless mean. |
| `model/src/bdh_surgery/ablate.py` | Scores each merged neuron by how strongly it responds to *both* parents' directions at once — the "collision set" that Act 4 ablates. |
| `model/src/bdh_surgery/damage.py` | Computes D: how much worse the merged model's loss is than the parent it's replacing, reported per direction (d_a, d_b) as well as pooled (d_mean), since pooling alone would hide the difference between the two target directions. |
| `model/src/bdh_surgery/sweep.py` | Runs the full θ × seed grid (11 θ values × 3 seeds = 33 rows) and writes `artifacts/runs.csv`. |
| `model/src/bdh_surgery/export.py` | Converts the trained tensors and sweep results into the static JSON + `.bin` files the browser reads. Ships int8 (symmetric, per-tensor scale) so all 11 swept θ fit in ~4.6 MB instead of ~18 MB at float32; the browser dequantizes once at load, so the forward pass never sees a quantized value. |
| `model/scripts/measure_locality.py` | Reproduces the locality table above: shells out to node and drives `bdh_forward.js`/`merge.js`/`collision.js` against `web/public/data/`, exactly as Act 4 does, for θ ∈ {0, 0.5, 1} × k ∈ {40, 100}, with 12 random draws so the control's spread is visible. Needs the export to exist and node on PATH, so it is not part of the pytest suite. |
| `model/scripts/probe_diagnostic.py` | Reproduces the 0.736 → 0.757 probe check: trains a base and both parents at a reduced step budget and computes M twice, once on the shared pivot probe and once with each parent on its own target-direction corpus. Not part of the pytest suite (it trains models). |
| `web/src/bdh_forward.js` | The recurrent BDH forward pass, ported to JavaScript so the browser can run real inference on the exported weights, checked against `recurrent.py` to a max absolute difference of 3.5e-07. |
| `web/src/merge.js` / `web/src/collision.js` | Client-side ports of `merge.py` / `ablate.py`, so the concat/average toggle and the collision heatmap update instantly with no server round trip. |
| `web/src/lib/Narrative.svelte` | The four-act guided walkthrough described below. |
| `web/src/lib/HonestyBadge.svelte` | Renders the `live` / `precomputed` / `illustration` labels applied throughout the UI. |
| `web/src/lib/Surgery.svelte` | Act 4's ablation UI: lets the learner ablate the collision set or an equally-sized random control and compare teacher-forced loss. |

## Dataset selector

A picker in the header switches between the three datasets described above
(baseline, long_phrase, large_vocab). Switching is a full reload — new parents, a
fresh collision analysis — since the datasets differ in `n_concepts` and, for
large_vocab, `vocab_size` too (40 concepts need a wider token layout than 24 fit in;
see `model/src/bdh_surgery/datasets.py`). θ itself carries over across the switch: all
three datasets export the same 11 θ keys, so staying on θ = 0.5 while changing dataset
is what lets you compare the same overlap level under two different vocabularies.

## The four acts

1. **Fuse** — the page opens already showing a fused preset (θ = 0.5, concat mode);
   there is no blank canvas or Run button. The merged neuron axis is drawn with the
   parent-A/parent-B split visible.
2. **Concat vs. average** — toggle merge mode and compare four columns of generation
   (parent A, parent B, merged, oracle target) on one shared prompt.
3. **The phase diagram** — D across the swept θ grid, plotted against θ by default (the
   strong relationship) with a toggle to plot against M instead (the flat one), and the
   averaging control as a second series sitting *above* concat; this is where the revised
   claim above is presented.
4. **Surgery** — ablate the collision set vs. a size-matched random control and read the
   *difference* in teacher-forced loss. In this artifact that difference is negative;
   the act is the honest demonstration of the retracted second half of the hypothesis,
   not a repair demo.

The narrative gates each act behind the one before it — Acts 2-4 are visually locked and
inert (using the HTML `inert` attribute, not just styling) until the learner has stepped
through their introduction — and unlocks all four for free exploration once the walkthrough
finishes or is explicitly skipped.

## Live / precomputed / synthetic / illustration

| Kind | Applied to | What it means |
|---|---|---|
| **live** | Output panels (Act 2), the merged neuron strip (Act 1), collision ablation and its loss stats (Act 4) | Computed in your browser right now, via `bdh_forward.js`/`merge.js`/`collision.js`, from the weight tensors that were loaded over the network. Change θ, merge mode, or the ablation set and these recompute immediately. |
| **precomputed** | The phase diagram (Act 3), the parent weights themselves (Act 1) | The phase diagram is `sweep.json`, produced offline by `sweep.py`/`export.py` — the browser only reads it and never reruns the sweep. The parent weight *tensors* are likewise trained offline and shipped as static `.bin` files; what the browser does live is the forward pass and merge *over* those weights, not the training. |
| **synthetic** | All training and evaluation data (`domains.py`) | Two artificial languages over a shared pivot vocabulary, generated procedurally — not natural language, not Europarl, not scraped from anywhere. |
| **illustration** | The θ-overlap schematic in Act 1's narrative | A plain conceptual diagram (two overlapping circles), explicitly labelled as not a rendering of measured data. |

A persistent footer at the bottom of the page restates the scale caps on every view:
n = 1024 per parent, 2048 merged (under concat), sequence length 16, and the words
"toy-scale reimplementation — not an official BDH model."

## Known limitations

1. **Toy scale.** Our models are n = 1024, 208,896 parameters per parent (merged:
   n = 2048, 405,504 parameters). The paper's BDH-GPU models were n = 24,576, 19M
   parameters per parent (merged n = 49,152, 38M). Our numbers do not speak for
   theirs — this is a labelled reimplementation, not an official BDH model, and the
   UI says so on every page.
2. **Synthetic domains, not natural language.** Two artificial languages over a shared
   pivot, not a real bilingual corpus.
3. **Ablation does not perfectly isolate a neuron.** `x_sparse` also feeds attention
   scores that sum over the entire neuron axis, so removing any neuron perturbs the
   attention distribution for every other neuron by roughly 2% of typical activation
   scale. This is intrinsic to attention, not a bug — it's exactly why the size-matched
   random-ablation control exists. The reportable result is always the *difference*
   between collision-ablation and random-ablation loss, never the raw recovery.
4. **M's flatness is the most important open question**, as described above: we cannot
   yet say whether representational overlap is genuinely non-predictive of
   mergeability, or whether the metric is mis-specified. The same caveat covers the
   collision score's failure — both metrics pool over the whole neuron axis.
5. **The random-ablation control in the browser is a single draw.** `Surgery.svelte`
   averages over 8 seeds and reports their spread; the offline
   `model/scripts/measure_locality.py` uses 12. At k = 40 the collision-vs-random gap
   is within that spread (SD ≈ 0.011–0.018), so only the k = 100 results separate from
   chance.
6. **No parameter-count control.** Comparing the merged 2n model against a single parent
   trained at 2n would rule out a pure size effect. It was planned, deferred, and never
   run; nothing in this repository claims otherwise.

## How to reproduce

```bash
# Python pipeline (from repo root)
cd model && uv sync && uv run pytest tests/        # 44 tests

cd model && uv run python -m bdh_surgery.sweep      # baseline, ~60 min -> artifacts/runs.csv

# The other two datasets, same grid, run the same way against their own spec:
cd model && uv run python -c "
from pathlib import Path
from bdh_surgery.sweep import run_sweep
from bdh_surgery.datasets import LONG_PHRASE, LARGE_VOCAB
for spec in (LONG_PHRASE, LARGE_VOCAB):
    run_sweep(Path('..')/'artifacts'/f'runs_{spec.id}.csv', spec=spec)
"                                                    # ~60 min each

cd model && uv run python -m bdh_surgery.export     # writes web/public/data/, all 3 datasets

# The two claim-checking diagnostics (neither is part of the pytest suite; baseline only)
cd model && uv run python scripts/measure_locality.py   # the locality table above; needs node
cd model && uv run python scripts/probe_diagnostic.py   # M under pivot vs. own-direction probes

# Web front end (from repo root)
cd web && npm install && npm run dev                # dev server
cd web && npm run build                             # static build -> web/dist/
```

The weights and `sweep.json`/`probes.json`/`manifest.json` committed under
`web/public/data/` were produced by exactly this sequence — running it again with the
same seeds (`SEEDS = (0, 1, 2)` in `model/src/bdh_surgery/sweep.py`) reproduces them for
all three datasets. `manifest.json` is schema v2: everything is nested under
`datasets.<id>` (`baseline` | `long_phrase` | `large_vocab`), each with its own
`config`, `n_concepts`, and `featured` weights — see `model/src/bdh_surgery/export.py`.

## Credits and licences

- **BDH model code** (`model/src/bdh_surgery/bdh.py`): adapted from
  [pathwaycom/bdh](https://github.com/pathwaycom/bdh), MIT licence (confirmed via the
  GitHub API, not assumed). Copyright header retained. See `PROVENANCE.md` for the full
  provenance record, including the single documented deviation from upstream behaviour.
- **This repository's own code and documentation**: implemented with substantial AI
  assistance (Claude), from a human-authored plan and specification, with every task
  reviewed. See `PROVENANCE.md`.
- **This repository's licence**: see `LICENSE` (Apache License 2.0) at the repo root.
- **Paper referenced**: arXiv:2509.26507, "Baby Dragon Hatchling" (Pathway), §7.1.

## Layout

- `model/` — Python (uv). Trains two toy BDH-GPU models on synthetic domains, merges
  them per the paper's concatenation rule, exports weights for the browser.
  See `model/README.md`.
- `web/` — Svelte + Vite static site. Runs the BDH-GPU forward pass and the merge
  sandbox client-side against the exported weights.
- `docs/` — presentation decks and the spec. **Outstanding deliverable:** the one-page
  concept summary (`docs/concept-summary.pdf`) required by the problem statement is
  **not yet written** — see `docs/README.md`. It has to be authored by the team; nothing
  in this repository generates it.
- `ps/` — the problem statement PDFs (`Pathway PS.pdf`, `Pathway PS_revised.pdf`) and the
  primary-source paper PDFs (`2509.26507v1.pdf`, `bdh-cq-2608.09888.pdf`). Not part of
  the submission.
- `PROVENANCE.md` — source, licence, and AI-assistance disclosure.
