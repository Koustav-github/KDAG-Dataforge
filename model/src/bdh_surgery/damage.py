"""Merge damage: how much worse the merged model is than the parent it replaces.

Reported per direction as well as pooled. The paper's headline finding is an
asymmetry (into-pivot survives, out-of-pivot degrades); pooling would hide it.
"""
from dataclasses import dataclass

from .bdh import BDH
from .domains import BOS_A, BOS_B, make_corpus
from .train import EVAL_SEQS, evaluate


@dataclass
class Damage:
    d_a: float
    d_b: float
    d_mean: float
    loss_merged_a: float
    loss_merged_b: float
    loss_parent_a: float
    loss_parent_b: float


def measure_damage(merged: BDH, parent_a: BDH, parent_b: BDH,
                   lex_a: list[int], lex_b: list[int], seed: int) -> Damage:
    eval_a = make_corpus(lex_a, BOS_A, EVAL_SEQS, seed=seed + 10_000)
    eval_b = make_corpus(lex_b, BOS_B, EVAL_SEQS, seed=seed + 20_000)
    ma, mb = evaluate(merged, eval_a), evaluate(merged, eval_b)
    pa, pb = evaluate(parent_a, eval_a), evaluate(parent_b, eval_b)
    return Damage(ma - pa, mb - pb, ((ma - pa) + (mb - pb)) / 2, ma, mb, pa, pb)
