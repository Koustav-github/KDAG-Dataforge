# DataForge 2026 — Pathway Track

Team Invariance, IIT Kharagpur. Submission for the Pathway track: an
interactive explainer connecting **model merging / composability in BDH**
(arXiv:2509.26507 §7.1) to the concept of parametric memory in LLMs. See
`ps/DataForge_Two_Proposals.docx` for the full design rationale.

## Layout

- `model/` — Python (uv). Trains two toy BDH-GPU models on synthetic domains,
  merges them per the paper's concatenation rule, exports weights for the
  browser. See `model/README.md`.
- `web/` — static site, no build step. Runs the BDH-GPU forward pass and the
  merge sandbox client-side against the exported weights.
- `docs/` — the one-page concept summary (PDF), required by the problem
  statement.
- `ps/` — problem statement and primary-source PDFs (not part of the
  submission).
- `PROVENANCE.md` — source, license, and AI-assistance disclosure.

## Status

Skeleton only; pipeline and sandbox are not yet implemented.
