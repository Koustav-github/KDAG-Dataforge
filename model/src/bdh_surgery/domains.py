import random

import torch

PAD, BOS_A, BOS_B, SEP, EOS = 0, 73, 74, 75, 76
N_CONCEPTS = 24
PIVOT_BASE = 1          # pivot tokens occupy 1..24
TARGET_BASE = 25        # target surface tokens occupy 25..72
PHRASE_LEN = 3
SEQ_LEN = 16


def build_lexicons(theta: float, seed: int) -> tuple[list[int], list[int]]:
    """Two surface lexicons sharing a theta fraction of concepts.

    Concepts are indexed 0..N_CONCEPTS-1. Shared concepts get an identical
    surface token in both languages; the rest get distinct tokens.
    """
    assert 0.0 <= theta <= 1.0
    rng = random.Random(seed)
    n_shared = round(theta * N_CONCEPTS)
    shared = set(rng.sample(range(N_CONCEPTS), n_shared))
    pool = list(range(TARGET_BASE, TARGET_BASE + 2 * N_CONCEPTS))
    rng.shuffle(pool)
    lex_a, lex_b, cursor = [], [], 0
    for concept in range(N_CONCEPTS):
        if concept in shared:
            tok = pool[cursor]; cursor += 1
            lex_a.append(tok); lex_b.append(tok)
        else:
            lex_a.append(pool[cursor]); lex_b.append(pool[cursor + 1]); cursor += 2
    return lex_a, lex_b


def measured_lexicon_overlap(lex_a: list[int], lex_b: list[int]) -> float:
    return sum(1 for a, b in zip(lex_a, lex_b) if a == b) / len(lex_a)


def _pad(row: list[int]) -> list[int]:
    return row[:SEQ_LEN] + [PAD] * max(0, SEQ_LEN - len(row))


def make_corpus(lex: list[int], bos: int, n_seqs: int, seed: int,
                phrase_len: int = PHRASE_LEN) -> torch.Tensor:
    """Translation sequences: BOS p... SEP t... EOS, padded to SEQ_LEN."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_seqs):
        concepts = [rng.randrange(N_CONCEPTS) for _ in range(phrase_len)]
        row = ([bos] + [PIVOT_BASE + c for c in concepts]
               + [SEP] + [lex[c] for c in concepts] + [EOS])
        rows.append(_pad(row))
    return torch.tensor(rows, dtype=torch.int64)


def make_pivot_corpus(n_seqs: int, seed: int,
                      phrase_len: int = PHRASE_LEN) -> torch.Tensor:
    """Pivot-only sequences for base pretraining. Theta-independent."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n_seqs):
        concepts = [rng.randrange(N_CONCEPTS) for _ in range(2 * phrase_len)]
        rows.append(_pad([BOS_A] + [PIVOT_BASE + c for c in concepts] + [EOS]))
    return torch.tensor(rows, dtype=torch.int64)
