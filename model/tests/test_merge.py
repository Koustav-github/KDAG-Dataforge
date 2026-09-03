import pytest


# Rule under test: concatenate tensors with an n dimension, average the rest
# (arXiv:2509.26507 SS7.1 / Appendix B.4).
@pytest.mark.skip(reason="TODO: merge.py not implemented yet")
def test_merge_concat_matches_paper_rule():
    raise NotImplementedError
