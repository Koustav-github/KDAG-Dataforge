"""Representational overlap M between two trained parents.

BDH neurons are sparse and reported as near-monosemantic (§6.2-6.4), so a
matching-based measure is the natural choice over a subspace measure like CKA:
we ask which of A's neurons have a counterpart in B, not whether the two span
a similar space.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .bdh import BDH, neuron_axis

TAUS = (0.5, 0.7, 0.9)


@torch.no_grad()
def activation_profiles(net: BDH, probe: torch.Tensor) -> np.ndarray:
    """Each neuron's sparse activation y over every probe token. Shape (n, tokens)."""
    net.eval()
    x = net.ln(net.embed(probe).unsqueeze(1))
    acts = []
    for _ in range(net.config.n_layer):
        x_sparse = F.relu(x @ net.encoder)
        y_kv = net.ln(net.attn(Q=x_sparse, K=x_sparse, V=x))
        y_sparse = F.relu(y_kv @ net.encoder_v)
        xy = x_sparse * y_sparse
        acts.append(xy)
        y = net.ln(xy.transpose(1, 2).reshape(x.shape[0], 1, probe.shape[1], -1)
                   @ net.decoder)
        x = net.ln(x + y)
    stacked = torch.stack(acts).mean(0)                       # average over layers
    n = neuron_axis(net.config)
    # reshape(-1, n) collapses (nh, D, T) assuming nh == 1, which holds project-wide.
    return stacked.reshape(-1, n).T.cpu().numpy()             # (n, tokens)


def _zscore(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(axis=1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def matched_correlations(net_a: BDH, net_b: BDH, probe: torch.Tensor) -> np.ndarray:
    """Hungarian-matched per-neuron correlations, sorted descending."""
    a, b = _zscore(activation_profiles(net_a, probe)), _zscore(activation_profiles(net_b, probe))
    corr = a @ b.T                                            # (n, n) cosine similarity
    rows, cols = linear_sum_assignment(-corr)                 # maximise total correlation
    return np.sort(corr[rows, cols])[::-1]


def overlap_at(matched: np.ndarray, tau: float) -> float:
    """M = fraction of matched neuron pairs correlating above tau."""
    return float((matched >= tau).mean())


def mean_matched_correlation(matched: np.ndarray) -> float:
    """Thresholdless fallback, used if matched counting proves noisy at n=1024."""
    return float(matched.mean())
