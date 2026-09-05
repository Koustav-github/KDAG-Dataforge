import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.merge import (
    CONCAT_TENSORS, AVERAGE_TENSORS, merged_config, merge_concat, merge_average,
)


def test_every_parameter_is_classified_exactly_once():
    names = {k for k, _ in BDH(TOY).named_parameters()}
    classified = set(CONCAT_TENSORS) | set(AVERAGE_TENSORS)
    assert names - {"attn.freqs"} <= classified
    assert set(CONCAT_TENSORS).isdisjoint(AVERAGE_TENSORS)


def test_concat_axis_is_the_neuron_axis_for_every_concat_tensor():
    net, n = BDH(TOY), neuron_axis(TOY)
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    for name, axis in CONCAT_TENSORS.items():
        assert tensors[name].shape[axis] == n, f"{name} axis {axis} is not the n axis"


def test_average_tensors_have_no_neuron_axis():
    net, n = BDH(TOY), neuron_axis(TOY)
    tensors = dict(net.named_parameters())
    for name in AVERAGE_TENSORS:
        assert n not in tuple(tensors[name].shape)


def test_merge_concat_doubles_n_and_matches_param_formula():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_concat(a, b)
    assert neuron_axis(m.config) == 2 * neuron_axis(TOY) == 2048
    assert sum(p.numel() for p in m.parameters()) == 3 * 2048 * 64 + 2 * 96 * 64 == 405504


def test_merge_concat_places_parents_side_by_side():
    a, b = BDH(TOY), BDH(TOY)
    m, n = merge_concat(a, b), neuron_axis(TOY)
    assert torch.equal(m.decoder[:n], a.decoder)
    assert torch.equal(m.decoder[n:], b.decoder)
    assert torch.equal(m.encoder[..., :n], a.encoder)
    assert torch.equal(m.encoder[..., n:], b.encoder)


def test_merge_averages_the_non_neuron_tensors():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_concat(a, b)
    assert torch.allclose(m.lm_head, (a.lm_head + b.lm_head) / 2)
    assert torch.allclose(m.embed.weight, (a.embed.weight + b.embed.weight) / 2)


def test_merge_average_keeps_size_and_blends_all_tensors():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_average(a, b)
    assert neuron_axis(m.config) == neuron_axis(TOY)
    ta = dict(a.named_parameters()) | {"attn.freqs": a.attn.freqs}
    tb = dict(b.named_parameters()) | {"attn.freqs": b.attn.freqs}
    tm = dict(m.named_parameters()) | {"attn.freqs": m.attn.freqs}
    for name in CONCAT_TENSORS:
        assert torch.allclose(tm[name], (ta[name] + tb[name]) / 2), name
    for name in AVERAGE_TENSORS:
        assert torch.allclose(tm[name], (ta[name] + tb[name]) / 2), name


def test_both_merges_run_a_forward_pass():
    a, b = BDH(TOY), BDH(TOY)
    idx = torch.randint(0, TOY.vocab_size, (1, 12))
    for m in (merge_concat(a, b), merge_average(a, b)):
        logits, _ = m(idx)
        assert logits.shape == (1, 12, TOY.vocab_size)
        assert torch.isfinite(logits).all()
