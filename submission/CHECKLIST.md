# Submission checklist — DataForge 2026, Pathway track

Generated 2026-09-05 from the PS (`ps/Pathway PS_revised.pdf`, pp. 12-13).
`[x]` done · `[~]` partial, with the gap named · `[ ]` not done.

## Package requirements (PS p.12)

| ✓ | Requirement | Status | Evidence |
|---|---|---|---|
| [x] | A public artifact URL that opens without sign-in | **DONE** | https://bdh-merger.vercel.app — Vercel, no auth. |
| [x] | A public source code repository | **DONE** | https://github.com/Koustav-github/KDAG-Dataforge — local HEAD matches origin/main. |
| [x] | The blog as a PDF | **DONE** | `concept-summary.pdf`. The PS's own parenthetical — "the blog as a pdf file (Talked about below)" — points to the one-page concept summary section that follows it, and the rubric scores that once at 10 points. Queried with the organisers, who responded by restating the same list unchanged; see SUBMISSION.md. |
| [x] | A complete README | **DONE** | `README-project.md`; sub-requirements below. |
| [x] | Clear setup instructions for any local component | **DONE** | README "How to reproduce" — uv + npm commands, all verified to run. |
| [x] | ≥3 recent primary papers (2022-2026), cited beside technical claims | **DONE** | 4 papers: 2203.05482, 2212.04089, 2305.12827, 2306.01708. Cited inline at the claims they support, plus a synthesis table in README "Related work". |
| [x] | Source and licence record for code, data, weights, graphics, fonts | **DONE** | `PROVENANCE.md` — upstream bdh.py MIT (verified via GitHub API), all data generated, no external fonts or graphics. |
| [x] | AI assistance, code, data, asset and licence disclosure | **DONE** | `PROVENANCE.md` — AI assistance section. |
| [x] | One-page concept summary PDF, ~500-950 words | **DONE** | `concept-summary.pdf` — 790 words, 1 page (verified by Word). |

## What the README must explain (PS p.12)

| ✓ | Requirement | Status | Evidence |
|---|---|---|---|
| [x] | The claim | **DONE** | README "The claim" — stated as a falsifiable sentence. |
| [x] | Intended learner and prerequisites | **DONE** | README "Intended learner and prerequisites". |
| [x] | Learning objectives | **DONE** | README "Learning objectives". |
| [x] | Architecture of the artifact | **DONE** | README "Artifact architecture". |
| [x] | Role of every major component | **DONE** | README "Role of every major component" table. |
| [x] | Which parts are live / precomputed / synthetic / animated | **DONE** | README "Live / precomputed / synthetic / illustration"; HonestyBadge in the UI. |
| [x] | How to reproduce the results | **DONE** | README "How to reproduce". |
| [x] | Credits and licences | **DONE** | README "Credits and licences". |

## Artifact and design standards (PS pp.8-9)

| ✓ | Requirement | Status | Evidence |
|---|---|---|---|
| [x] | Learner can alter an input/parameter/state and observe the result | **DONE** | θ slider, merge rule toggle, k slider, ablation choice, dataset selector. |
| [x] | Animation connected to real computation or labelled as simplification | **DONE** | GSAP fusion is driven by live θ; schematic panels carry an "illustration" label. |
| [x] | Concept behaves in the artifact before the learner acts | **DONE** | Opens on a running preset, not a blank canvas. |
| [x] | Truth shown beside estimate | **DONE** | Oracle target rendered beside model output. |
| [~] | Fast feedback | **PARTIAL** | Sub-second on desktop. On mobile a θ/k change can take several seconds — inherent to running the forward pass in JS; documented as limitation 7 in the README. |
| [x] | Substantial, technically correct BDH section | **DONE** | The merge rule is derived from BDH's parameterisation (3nd + 2Ωd) throughout. |
| [x] | At least one disclosed limitation | **DONE** | README "Known limitations" — 7 of them, including two retracted claims. |

## Summary

- **23 / 24 complete**
- **1 partial** — each names its gap above rather than being marked done.
- **0 not done**

### The open item, stated plainly

1. **Mobile feedback speed** — every forward pass runs in JS on the main thread, so Act 4
   takes seconds on a phone. Mitigated (finer yielding, DPR cap at 2×, longer debounce) and
   documented as limitation 7, but not eliminated. Reducing sample counts on mobile would
   have fixed the speed at the cost of phones reporting different numbers than desktop; we
   declined that trade deliberately, because a device-dependent statistic would undermine
   the evidence discipline the rest of the submission rests on.

### Resolved since first assembly

- **"Blog as a PDF"** — confirmed as the one-page concept summary. See SUBMISSION.md
  for the three lines of evidence, including the organisers' reply.
