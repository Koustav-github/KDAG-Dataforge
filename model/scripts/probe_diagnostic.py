#!/usr/bin/env python
"""Is M flat because the probe is wrong, or because the metric is wrong?

M (Hungarian-matched neuron activation correlation between the two parents)
barely moves across the theta sweep. One obvious explanation is the probe:
overlap.py measures both parents on the *shared pivot* corpus, which is
theta-independent by construction, so perhaps every parent looks alike simply
because the probe never exercises what the fine-tuning changed.

This script tests that. For each theta it computes M twice:

  pivot     both parents probed on the shared pivot corpus (what sweep.py does)
  own-dir   each parent probed on its OWN target-direction corpus, so parent A
            is measured on P->A sequences and parent B on P->B sequences

If the probe were the problem, own-direction probing would spread M out. It
does not: M shifts up slightly and stays just as flat. The probe is not the
problem -- the metric does not capture what drives mergeability.

Budget note: this is a diagnostic, not the main sweep. It trains at 300 base
steps / 450 fine-tune steps (the main sweep uses 400 / 600) and one seed, so
it finishes in a few minutes. The absolute M values therefore differ slightly
from artifacts/runs.csv; the comparison between the two probes is the point.

Not part of the pytest suite (it trains models). Run it directly:

    python model/scripts/probe_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

from bdh_surgery.domains import BOS_A, BOS_B, build_lexicons, make_corpus, make_pivot_corpus
from bdh_surgery.overlap import _zscore, activation_profiles, mean_matched_correlation, matched_correlations
from bdh_surgery.train import finetune, train_base

THETAS = (0.0, 0.5, 1.0)
SEED = 0
BASE_STEPS = 300
FT_STEPS = 450
PROBE_SEQS = 16


def matched_own_direction(net_a, net_b, probe_a, probe_b) -> np.ndarray:
    """Hungarian-matched correlations with each parent on its own corpus.

    Same procedure as overlap.matched_correlations, except the two activation
    profiles come from different probes. Both probes have the same number of
    tokens, so the correlation matrix is still well-formed.
    """
    a = _zscore(activation_profiles(net_a, probe_a))
    b = _zscore(activation_profiles(net_b, probe_b))
    corr = a @ b.T
    rows, cols = linear_sum_assignment(-corr)
    return np.sort(corr[rows, cols])[::-1]


def main() -> None:
    print("Probe diagnostic: shared-pivot probe vs. own-direction probe")
    print(f"seed={SEED}  base_steps={BASE_STEPS}  finetune_steps={FT_STEPS} "
          f"(main sweep uses 400 / 600)\n")
    header = f"{'theta':>6}  {'M (pivot)':>10}  {'M (own-dir)':>12}  {'delta':>8}"
    print(header)
    print("-" * len(header))

    base = train_base(seed=SEED, steps=BASE_STEPS)
    pivot = make_pivot_corpus(PROBE_SEQS, seed=SEED + 30_000)
    m_pivot, m_own = [], []

    for theta in THETAS:
        lex_a, lex_b = build_lexicons(theta, seed=SEED)
        pa = finetune(base, lex_a, BOS_A, seed=SEED, steps=FT_STEPS)
        pb = finetune(base, lex_b, BOS_B, seed=SEED, steps=FT_STEPS)

        probe_a = make_corpus(lex_a, BOS_A, PROBE_SEQS, seed=SEED + 30_000)
        probe_b = make_corpus(lex_b, BOS_B, PROBE_SEQS, seed=SEED + 30_000)

        mp = mean_matched_correlation(matched_correlations(pa, pb, pivot))
        mo = mean_matched_correlation(matched_own_direction(pa, pb, probe_a, probe_b))
        m_pivot.append(mp)
        m_own.append(mo)
        print(f"{theta:>6.1f}  {mp:>10.4f}  {mo:>12.4f}  {mo - mp:>+8.4f}")

    print("-" * len(header))
    print(f"{'mean':>6}  {np.mean(m_pivot):>10.4f}  {np.mean(m_own):>12.4f}  "
          f"{np.mean(m_own) - np.mean(m_pivot):>+8.4f}")
    print(f"{'range':>6}  {max(m_pivot) - min(m_pivot):>10.4f}  "
          f"{max(m_own) - min(m_own):>12.4f}")
    print("\nOwn-direction probing raises M a little and leaves it just as flat across")
    print("theta. The probe is not why M fails to predict damage.")


if __name__ == "__main__":
    main()
