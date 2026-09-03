# Merge rule from arXiv:2509.26507 SS7.1: concatenate every tensor with an n
# dimension (Dy, Dx, E, RoPE frequency buffers) along n; average everything else
# (token embeddings, token-prediction weights). No retraining after merge.
# TODO: implement against the trained parent pair; test_merge.py checks this rule.


def merge_concat(model_a, model_b):
    raise NotImplementedError


def merge_average(model_a, model_b):
    """The Transformer-style baseline the sandbox contrasts concat against."""
    raise NotImplementedError
