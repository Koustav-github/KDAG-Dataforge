import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.recurrent import recurrent_logits


def test_recurrent_matches_parallel():
    torch.manual_seed(0)
    net = BDH(TOY).eval()
    idx = torch.randint(0, TOY.vocab_size, (1, 24))
    with torch.no_grad():
        parallel, _ = net(idx)
    recurrent = recurrent_logits(net, idx)
    assert recurrent.shape == parallel.shape
    assert (parallel - recurrent).abs().max().item() < 1e-4
    assert torch.equal(parallel.argmax(-1), recurrent.argmax(-1))
