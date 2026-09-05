import copy
import csv
from functools import lru_cache
from pathlib import Path

from .damage import measure_damage
from .domains import BOS_A, BOS_B, build_lexicons, make_pivot_corpus
from .merge import merge_average, merge_concat
from .overlap import TAUS, matched_correlations, mean_matched_correlation, overlap_at
from .train import finetune, train_base

THETAS = tuple(round(i / 10, 1) for i in range(11))
SEEDS = (0, 1, 2)
PROBE_SEQS = 16

FIELDS = ["theta", "seed", "m_tau50", "m_tau70", "m_tau90", "m_mean",
          "d_a", "d_b", "d_mean", "d_avg_arm", "loss_parent_a", "loss_parent_b",
          "self_merge_d", "params_merged"]


@lru_cache(maxsize=None)
def _base(seed: int, steps: int):
    return train_base(seed=seed, steps=steps)


def run_pair(theta: float, seed: int, steps_base: int = 400,
             steps_ft: int = 600) -> dict:
    base = _base(seed, steps_base)
    lex_a, lex_b = build_lexicons(theta, seed=seed)
    pa = finetune(base, lex_a, BOS_A, seed=seed, steps=steps_ft)
    pb = finetune(base, lex_b, BOS_B, seed=seed, steps=steps_ft)

    probe = make_pivot_corpus(PROBE_SEQS, seed=seed + 30_000)
    matched = matched_correlations(pa, pb, probe)

    concat = merge_concat(pa, pb)
    average = merge_average(pa, pb)
    d_concat = measure_damage(concat, pa, pb, lex_a, lex_b, seed)
    d_average = measure_damage(average, pa, pb, lex_a, lex_b, seed)
    d_self = measure_damage(merge_concat(pa, copy.deepcopy(pa)), pa, pa,
                            lex_a, lex_a, seed)

    return {
        "theta": theta, "seed": seed,
        "m_tau50": overlap_at(matched, TAUS[0]),
        "m_tau70": overlap_at(matched, TAUS[1]),
        "m_tau90": overlap_at(matched, TAUS[2]),
        "m_mean": mean_matched_correlation(matched),
        "d_a": d_concat.d_a, "d_b": d_concat.d_b, "d_mean": d_concat.d_mean,
        "d_avg_arm": d_average.d_mean,
        "loss_parent_a": d_concat.loss_parent_a,
        "loss_parent_b": d_concat.loss_parent_b,
        "self_merge_d": d_self.d_mean,
        "params_merged": sum(p.numel() for p in concat.parameters()),
    }


def run_sweep(out_csv: Path, thetas=THETAS, seeds=SEEDS, **kw) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for theta in thetas:
            for seed in seeds:
                row = run_pair(theta, seed, **kw)
                writer.writerow(row)
                fh.flush()
                print(f"theta={theta:.1f} seed={seed} "
                      f"M={row['m_mean']:.3f} D={row['d_mean']:+.4f}", flush=True)
    return out_csv


if __name__ == "__main__":
    run_sweep(Path(__file__).resolve().parents[3] / "artifacts" / "runs.csv")
