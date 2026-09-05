"""Collision neurons: units in the merged model that respond to both target
languages. A neuron inherited cleanly from one parent should be selective for
that parent's direction; one that fires for both is where the two models
compete for the same output channel.
"""
import copy

import numpy as np
import torch

from .bdh import BDH
from .overlap import activation_profiles


def collision_scores(merged: BDH, probe_a: torch.Tensor,
                     probe_b: torch.Tensor) -> np.ndarray:
    """High where a neuron is strongly and equally active on both directions."""
    a = activation_profiles(merged, probe_a).mean(axis=1)
    b = activation_profiles(merged, probe_b).mean(axis=1)
    joint = np.minimum(a, b)                     # active on both
    imbalance = np.abs(a - b) / (a + b + 1e-8)   # 0 when equally active
    return joint * (1.0 - imbalance)


def top_collision_neurons(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:k].copy()


def random_neurons(n_total: int, k: int, seed: int) -> np.ndarray:
    """Size-matched control. Without this, 'we ablated and it improved' proves nothing."""
    return np.random.default_rng(seed).choice(n_total, size=k, replace=False)


def ablate(merged: BDH, indices: np.ndarray) -> BDH:
    """Zero the named neurons everywhere they appear on the n axis. Returns a copy."""
    out = copy.deepcopy(merged).eval()
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    with torch.no_grad():
        out.decoder[idx] = 0.0          # (n, D)
        out.encoder[..., idx] = 0.0     # (nh, D, n)
        out.encoder_v[..., idx] = 0.0   # (nh, D, n)
    return out
