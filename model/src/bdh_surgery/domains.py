import random

import torch

PAD, BOS_A, BOS_B, SEP, EOS = 0, 73, 74, 75, 76
N_CONCEPTS = 24
PIVOT_BASE = 1          # pivot tokens occupy 1..24
TARGET_BASE = 25        # target surface tokens occupy 25..72
PHRASE_LEN = 3
SEQ_LEN = 16


def layout(n_concepts: int) -> dict:
    """Token-id ranges for a vocabulary of `n_concepts` concepts.

    The original constants above hard-code the 24-concept case. Anything larger
    overflows them — at 40 concepts the target pool alone needs 80 ids and would
    run straight through BOS_A..EOS. So the ranges are computed instead:

        PAD 0 | pivot [1, 1+n) | target [1+n, 1+3n) | BOS_A BOS_B SEP EOS

    The target pool is 2n wide because a concept that is NOT shared between the
    two languages needs a distinct surface token in each.

    At n_concepts=24 this reproduces the module constants exactly, which is what
    makes the parametrised path safe to add underneath the existing one.
    """
    pivot_base = 1
    target_base = pivot_base + n_concepts
    bos_a = target_base + 2 * n_concepts
    return {
        "n_concepts": n_concepts,
        "pivot_base": pivot_base,
        "target_base": target_base,
        "bos_a": bos_a,
        "bos_b": bos_a + 1,
        "sep": bos_a + 2,
        "eos": bos_a + 3,
        "min_vocab": bos_a + 4,
    }


def build_lexicons(theta: float, seed: int, n_concepts: int = N_CONCEPTS,
                   target_base: int = TARGET_BASE) -> tuple[list[int], list[int]]:
    """Two surface lexicons sharing a theta fraction of concepts.

    Concepts are indexed 0..N_CONCEPTS-1. Shared concepts get an identical
    surface token in both languages; the rest get distinct tokens.
    """
    assert 0.0 <= theta <= 1.0
    rng = random.Random(seed)
    n_shared = round(theta * n_concepts)
    shared = set(rng.sample(range(n_concepts), n_shared))
    pool = list(range(target_base, target_base + 2 * n_concepts))
    rng.shuffle(pool)
    lex_a, lex_b, cursor = [], [], 0
    for concept in range(n_concepts):
        if concept in shared:
            tok = pool[cursor]; cursor += 1
            lex_a.append(tok); lex_b.append(tok)
        else:
            lex_a.append(pool[cursor]); lex_b.append(pool[cursor + 1]); cursor += 2
    return lex_a, lex_b


def measured_lexicon_overlap(lex_a: list[int], lex_b: list[int]) -> float:
    return sum(1 for a, b in zip(lex_a, lex_b) if a == b) / len(lex_a)


def _pad(row: list[int], seq_len: int = SEQ_LEN) -> list[int]:
    return row[:seq_len] + [PAD] * max(0, seq_len - len(row))


def make_corpus(lex: list[int], bos: int, n_seqs: int, seed: int,
                phrase_len: int = PHRASE_LEN, n_concepts: int = N_CONCEPTS,
                pivot_base: int = PIVOT_BASE, sep: int = SEP, eos: int = EOS,
                seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Translation sequences: BOS p... SEP t... EOS, padded to SEQ_LEN."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_seqs):
        concepts = [rng.randrange(n_concepts) for _ in range(phrase_len)]
        row = ([bos] + [pivot_base + c for c in concepts]
               + [sep] + [lex[c] for c in concepts] + [eos])
        rows.append(_pad(row, seq_len))
    return torch.tensor(rows, dtype=torch.int64)


def make_pivot_corpus(n_seqs: int, seed: int, phrase_len: int = PHRASE_LEN,
                      n_concepts: int = N_CONCEPTS, pivot_base: int = PIVOT_BASE,
                      bos: int = BOS_A, eos: int = EOS,
                      seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Pivot-only sequences for base pretraining. Theta-independent."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_seqs):
        concepts = [rng.randrange(n_concepts) for _ in range(2 * phrase_len)]
        rows.append(_pad([bos] + [pivot_base + c for c in concepts] + [eos], seq_len))
    return torch.tensor(rows, dtype=torch.int64)
