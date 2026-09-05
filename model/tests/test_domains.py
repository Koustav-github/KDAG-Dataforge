import torch
from bdh_surgery.domains import (
    N_CONCEPTS, SEQ_LEN, PAD, BOS_A, BOS_B, SEP, EOS,
    build_lexicons, make_corpus, make_pivot_corpus, measured_lexicon_overlap,
)


def test_theta_zero_gives_disjoint_lexicons():
    a, b = build_lexicons(0.0, seed=1)
    assert len(a) == len(b) == N_CONCEPTS
    assert set(a).isdisjoint(set(b))
    assert measured_lexicon_overlap(a, b) == 0.0


def test_theta_one_gives_identical_lexicons():
    a, b = build_lexicons(1.0, seed=1)
    assert a == b
    assert measured_lexicon_overlap(a, b) == 1.0


def test_theta_half_shares_half_the_concepts():
    a, b = build_lexicons(0.5, seed=1)
    shared = sum(1 for x, y in zip(a, b) if x == y)
    assert shared == N_CONCEPTS // 2
    assert measured_lexicon_overlap(a, b) == 0.5


def test_corpus_shape_and_structure():
    a, _ = build_lexicons(0.3, seed=1)
    corpus = make_corpus(a, BOS_A, n_seqs=32, seed=2)
    assert corpus.shape == (32, SEQ_LEN)
    assert corpus.dtype == torch.int64
    assert (corpus[:, 0] == BOS_A).all()
    assert (corpus == SEP).any(dim=1).all()
    assert (corpus == EOS).any(dim=1).all()


def test_corpus_is_deterministic_given_seed():
    a, _ = build_lexicons(0.3, seed=1)
    assert torch.equal(make_corpus(a, BOS_A, 16, seed=7), make_corpus(a, BOS_A, 16, seed=7))


def test_pivot_corpus_uses_no_target_tokens():
    corpus = make_pivot_corpus(n_seqs=16, seed=3)
    assert corpus.shape == (16, SEQ_LEN)
    allowed = {PAD, BOS_A, EOS} | set(range(1, N_CONCEPTS + 1))
    assert set(corpus.flatten().tolist()) <= allowed
    assert not ((corpus >= 25) & (corpus <= 72)).any()
