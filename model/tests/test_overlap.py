import copy
import numpy as np
import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.domains import make_pivot_corpus
from bdh_surgery.overlap import (
    TAUS, activation_profiles, matched_correlations,
    overlap_at, mean_matched_correlation,
)


def test_profiles_have_one_row_per_neuron():
    net, probe = BDH(TOY).eval(), make_pivot_corpus(8, seed=1)
    prof = activation_profiles(net, probe)
    assert prof.shape[0] == neuron_axis(TOY)
    assert prof.shape[1] == probe.shape[0] * probe.shape[1]


def test_identical_models_have_overlap_one():
    torch.manual_seed(0)
    net, probe = BDH(TOY).eval(), make_pivot_corpus(8, seed=1)
    matched = matched_correlations(net, copy.deepcopy(net), probe)
    assert matched.shape == (neuron_axis(TOY),)
    assert overlap_at(matched, 0.9) > 0.99
    assert mean_matched_correlation(matched) > 0.99


def test_independent_models_have_lower_overlap_than_identical_ones():
    torch.manual_seed(0); a = BDH(TOY).eval()
    torch.manual_seed(999); b = BDH(TOY).eval()
    probe = make_pivot_corpus(8, seed=1)
    assert (mean_matched_correlation(matched_correlations(a, b, probe))
            < mean_matched_correlation(matched_correlations(a, copy.deepcopy(a), probe)))


def test_overlap_at_is_monotone_in_tau():
    torch.manual_seed(0); a = BDH(TOY).eval()
    torch.manual_seed(7); b = BDH(TOY).eval()
    matched = matched_correlations(a, b, make_pivot_corpus(8, seed=1))
    vals = [overlap_at(matched, t) for t in TAUS]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in vals)
