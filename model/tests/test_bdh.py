import torch
from bdh_surgery.bdh import BDH, BDHConfig, TOY, neuron_axis


def test_toy_param_count_matches_3nd_plus_2vd():
    net = BDH(TOY)
    n, d, v = neuron_axis(TOY), TOY.n_embd, TOY.vocab_size
    assert neuron_axis(TOY) == 1024
    assert sum(p.numel() for p in net.parameters()) == 3 * n * d + 2 * v * d == 208896


def test_layernorm_is_parameter_free():
    assert sum(p.numel() for p in BDH(TOY).ln.parameters()) == 0


def test_forward_shapes_and_loss():
    net = BDH(TOY)
    idx = torch.randint(0, TOY.vocab_size, (2, 16))
    logits, loss = net(idx, targets=idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0
