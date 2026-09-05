import torch
import torch.nn.functional as F

from .bdh import BDH, Attention, neuron_axis


@torch.no_grad()
def recurrent_logits(net: BDH, idx: torch.Tensor) -> torch.Tensor:
    """Eq. (8) recurrent form. Equivalent to net(idx)[0]; O(N*D) state per token."""
    cfg = net.config
    assert cfg.n_head == 1, "recurrent form implemented for n_head=1 only"
    assert idx.shape[0] == 1, "recurrent form is batch-size-1 only"
    n, d, T = neuron_axis(cfg), cfg.n_embd, idx.shape[1]
    freqs = net.attn.freqs.view(n)
    rho = [torch.zeros(n, d) for _ in range(cfg.n_layer)]
    outs = []
    for t in range(T):
        x = net.ln(net.embed(idx[:, t : t + 1]).unsqueeze(1))          # (1,1,1,D)
        for layer in range(cfg.n_layer):
            x_sparse = F.relu(x @ net.encoder)                          # (1,1,1,N)
            phases = (t * freqs).view(1, 1, 1, n)
            qr = Attention.rope(phases, x_sparse).view(n)
            y_kv = (qr @ rho[layer]).view(1, 1, 1, d)                   # read before write
            rho[layer] = rho[layer] + torch.outer(qr, x.view(d))
            y_sparse = F.relu(net.ln(y_kv) @ net.encoder_v)
            y_mlp = (x_sparse * y_sparse).view(1, 1, 1, n) @ net.decoder
            x = net.ln(x + net.ln(y_mlp))
        outs.append(x.view(d))
    return torch.stack(outs).unsqueeze(0) @ net.lm_head
