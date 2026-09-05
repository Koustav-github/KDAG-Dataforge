import copy
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.domains import BOS_A, BOS_B, build_lexicons
from bdh_surgery.damage import Damage, measure_damage
from bdh_surgery.merge import merge_concat


def test_self_merge_of_identical_parents_reports_near_zero_damage():
    # The real self-merge control (sweep.py) concatenates a parent with a
    # bit-identical deepcopy of itself — not an independently seeded retrain —
    # and finds |D| <= 0.0035 across all 33 swept runs. Exercise the actual
    # merge_concat() call here rather than substituting the parent for the
    # merged model, so this test would catch a regression in the merge itself.
    torch.manual_seed(0)
    p = BDH(TOY).eval()
    lex_a, lex_b = build_lexicons(0.5, seed=0)
    merged = merge_concat(p, copy.deepcopy(p))
    d = measure_damage(merged, p, copy.deepcopy(p), lex_a, lex_b, seed=1)
    assert isinstance(d, Damage)
    assert all(map(torch.isfinite, map(torch.tensor, (d.d_a, d.d_b, d.d_mean))))
    assert abs(d.d_mean - (d.d_a + d.d_b) / 2) < 1e-9
    # Near-zero, matching the maximum-overlap floor measured in the real sweep.
    assert abs(d.d_a) < 0.05 and abs(d.d_b) < 0.05


def test_damage_is_zero_when_merged_equals_parents():
    torch.manual_seed(0)
    p = BDH(TOY).eval()
    lex_a, lex_b = build_lexicons(1.0, seed=0)
    d = measure_damage(copy.deepcopy(p), p, copy.deepcopy(p), lex_a, lex_b, seed=1)
    assert abs(d.d_a) < 1e-6 and abs(d.d_b) < 1e-6
