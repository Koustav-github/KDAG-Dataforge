import torch

from bdh_surgery.datasets import BASELINE, DATASETS, LARGE_VOCAB, LONG_PHRASE
from bdh_surgery.domains import (
    BOS_A, BOS_B, EOS, N_CONCEPTS, PIVOT_BASE, SEP, SEQ_LEN, TARGET_BASE,
    build_lexicons, layout, make_corpus, make_pivot_corpus,
)


def test_layout_at_24_reproduces_the_original_constants():
    """The parametrised layout must agree with the hard-coded constants at the
    baseline size, or the existing trained weights stop matching their data."""
    lay = layout(N_CONCEPTS)
    assert lay["pivot_base"] == PIVOT_BASE
    assert lay["target_base"] == TARGET_BASE
    assert (lay["bos_a"], lay["bos_b"], lay["sep"], lay["eos"]) == (BOS_A, BOS_B, SEP, EOS)


def test_every_dataset_fits_its_vocab_and_sequence_budget():
    for spec in DATASETS:
        spec.check()  # raises if not
        assert spec.config().vocab_size >= spec.layout["min_vocab"]


def test_token_ranges_never_overlap():
    for spec in DATASETS:
        lay = spec.layout
        pivot = range(lay["pivot_base"], lay["target_base"])
        target = range(lay["target_base"], lay["bos_a"])
        specials = {lay["bos_a"], lay["bos_b"], lay["sep"], lay["eos"]}
        assert set(pivot).isdisjoint(target)
        assert specials.isdisjoint(pivot)
        assert specials.isdisjoint(target)
        assert 0 not in pivot and 0 not in target and 0 not in specials  # PAD


def test_generated_corpora_stay_inside_the_declared_vocab():
    for spec in DATASETS:
        lay = spec.layout
        lex_a, lex_b = build_lexicons(0.5, seed=0, n_concepts=spec.n_concepts,
                                      target_base=lay["target_base"])
        assert len(lex_a) == len(lex_b) == spec.n_concepts
        for lex, bos in ((lex_a, lay["bos_a"]), (lex_b, lay["bos_b"])):
            corpus = make_corpus(lex, bos, 32, seed=1, phrase_len=spec.phrase_len,
                                 n_concepts=spec.n_concepts, pivot_base=lay["pivot_base"],
                                 sep=lay["sep"], eos=lay["eos"])
            assert corpus.shape == (32, SEQ_LEN)
            assert int(corpus.max()) < spec.vocab_size
        pivot = make_pivot_corpus(32, seed=2, phrase_len=spec.phrase_len,
                                  n_concepts=spec.n_concepts, pivot_base=lay["pivot_base"],
                                  bos=lay["bos_a"], eos=lay["eos"])
        assert int(pivot.max()) < spec.vocab_size


def test_larger_vocab_would_have_overflowed_the_old_fixed_layout():
    """Documents why the layout had to become parametric at all."""
    assert LARGE_VOCAB.layout["min_vocab"] > BASELINE.config().vocab_size


def test_long_phrase_changes_content_not_vocabulary():
    assert LONG_PHRASE.n_concepts == BASELINE.n_concepts
    assert LONG_PHRASE.layout == BASELINE.layout
    assert LONG_PHRASE.phrase_len > BASELINE.phrase_len


def test_baseline_spec_matches_the_untouched_defaults():
    """Baseline must go on producing byte-identical data to the already-trained
    weights, so its spec has to reduce to the module defaults exactly."""
    lay = BASELINE.layout
    a1, b1 = build_lexicons(0.4, seed=3)
    a2, b2 = build_lexicons(0.4, seed=3, n_concepts=BASELINE.n_concepts,
                            target_base=lay["target_base"])
    assert a1 == a2 and b1 == b2
    c1 = make_corpus(a1, BOS_A, 16, seed=5)
    c2 = make_corpus(a2, lay["bos_a"], 16, seed=5, phrase_len=BASELINE.phrase_len,
                     n_concepts=BASELINE.n_concepts, pivot_base=lay["pivot_base"],
                     sep=lay["sep"], eos=lay["eos"])
    assert torch.equal(c1, c2)
