import csv
from bdh_surgery.sweep import THETAS, SEEDS, run_pair, run_sweep

FAST = dict(steps_base=8, steps_ft=8)


def test_theta_grid_is_eleven_points_and_seeds_are_three():
    assert len(THETAS) == 11 and THETAS[0] == 0.0 and THETAS[-1] == 1.0
    assert len(SEEDS) == 3


def test_run_pair_returns_every_expected_column():
    row = run_pair(0.5, seed=0, **FAST)
    for key in ("theta", "seed", "m_tau50", "m_mean", "d_a", "d_b",
                "d_mean", "d_avg_arm", "self_merge_d", "params_merged"):
        assert key in row, f"missing column {key}"
    assert row["params_merged"] == 405504


def test_run_sweep_writes_one_row_per_pair(tmp_path):
    out = run_sweep(tmp_path / "runs.csv", thetas=(0.0, 1.0), seeds=(0, 1), **FAST)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    assert {float(r["theta"]) for r in rows} == {0.0, 1.0}
