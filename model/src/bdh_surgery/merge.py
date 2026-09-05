"""Merge rule from arXiv:2509.26507 §7.1.

Concatenate every tensor carrying the neuron dimension n along that axis;
average everything else. LayerNorm is parameter-free, so it needs no handling.
The RoPE frequency buffer is concatenated rather than recomputed for the new n,
exactly as the paper prescribes.
"""
import copy
import dataclasses

import torch

from .bdh import BDH, BDHConfig, neuron_axis

CONCAT_TENSORS: dict[str, int] = {
    "decoder": 0,       # (n, D)
    "encoder": 2,       # (nh, D, n)
    "encoder_v": 2,     # (nh, D, n)
    "attn.freqs": 3,    # (1, 1, 1, n) — buffer, not a parameter
}
AVERAGE_TENSORS: tuple[str, ...] = ("lm_head", "embed.weight")


def merged_config(cfg: BDHConfig) -> BDHConfig:
    return dataclasses.replace(
        cfg, mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier * 2)


def _blend(out: BDH, a: BDH, b: BDH, concat: bool) -> BDH:
    ta = dict(a.named_parameters()) | {"attn.freqs": a.attn.freqs}
    tb = dict(b.named_parameters()) | {"attn.freqs": b.attn.freqs}
    to = dict(out.named_parameters()) | {"attn.freqs": out.attn.freqs}
    with torch.no_grad():
        for name, axis in CONCAT_TENSORS.items():
            if concat:
                to[name].copy_(torch.cat([ta[name], tb[name]], dim=axis))
            else:
                to[name].copy_((ta[name] + tb[name]) / 2)
        for name in AVERAGE_TENSORS:
            to[name].copy_((ta[name] + tb[name]) / 2)
    return out.eval()


def merge_concat(a: BDH, b: BDH) -> BDH:
    """Grow along n: the merged model has n_a + n_b neurons, both parents intact."""
    assert a.config == b.config, "parents must share a config"
    return _blend(BDH(merged_config(a.config)), a, b, concat=True)


def merge_average(a: BDH, b: BDH) -> BDH:
    """The Transformer-style control: same size, representations blended."""
    assert a.config == b.config, "parents must share a config"
    return _blend(BDH(copy.deepcopy(a.config)), a, b, concat=False)
