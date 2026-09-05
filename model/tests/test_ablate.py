import numpy as np
import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.domains import BOS_A, BOS_B, build_lexicons, make_corpus
from bdh_surgery.merge import merge_concat
from bdh_surgery.ablate import (
    collision_scores, top_collision_neurons, random_neurons, ablate,
)


def _fixture():
    torch.manual_seed(0)
    m = merge_concat(BDH(TOY).eval(), BDH(TOY).eval())
    lex_a, lex_b = build_lexicons(0.5, seed=0)
    return (m, make_corpus(lex_a, BOS_A, 8, seed=1), make_corpus(lex_b, BOS_B, 8, seed=2))


def test_collision_scores_cover_the_merged_neuron_axis():
    m, pa, pb = _fixture()
    s = collision_scores(m, pa, pb)
    assert s.shape == (neuron_axis(m.config),) == (2048,)
    assert np.isfinite(s).all()


def test_top_collision_returns_k_unique_indices():
    m, pa, pb = _fixture()
    idx = top_collision_neurons(collision_scores(m, pa, pb), k=50)
    assert idx.shape == (50,) and len(set(idx.tolist())) == 50


def test_random_baseline_is_disjoint_size_matched_and_reproducible():
    a = random_neurons(2048, 50, seed=3)
    assert a.shape == (50,) and len(set(a.tolist())) == 50
    assert np.array_equal(a, random_neurons(2048, 50, seed=3))
    assert not np.array_equal(a, random_neurons(2048, 50, seed=4))


def test_ablate_zeroes_only_the_named_neurons_and_leaves_merged_untouched():
    m, pa, pb = _fixture()
    before_decoder = m.decoder.clone()
    before_encoder = m.encoder.clone()
    before_encoder_v = m.encoder_v.clone()
    idx = np.array([0, 5, 9])
    out = ablate(m, idx)
    assert torch.equal(m.decoder, before_decoder), "ablate must not mutate its input"
    assert torch.equal(m.encoder, before_encoder), "ablate must not mutate its input"
    assert torch.equal(m.encoder_v, before_encoder_v), "ablate must not mutate its input"
    assert out.decoder[idx].abs().sum().item() == 0
    assert out.encoder[..., idx].abs().sum().item() == 0
    assert out.encoder_v[..., idx].abs().sum().item() == 0
    keep = np.setdiff1d(np.arange(neuron_axis(m.config)), idx)
    assert torch.equal(out.decoder[keep], before_decoder[keep])
    assert torch.equal(out.encoder[..., keep], before_encoder[..., keep])
    assert torch.equal(out.encoder_v[..., keep], before_encoder_v[..., keep])


def test_ablated_model_still_runs():
    m, pa, pb = _fixture()
    out = ablate(m, top_collision_neurons(collision_scores(m, pa, pb), k=20))
    logits, _ = out(torch.randint(0, TOY.vocab_size, (1, 12)))
    assert torch.isfinite(logits).all()
