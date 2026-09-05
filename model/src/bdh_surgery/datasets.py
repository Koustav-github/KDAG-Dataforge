"""The synthetic translation scenarios the sweep runs over.

Every dataset uses the SAME protocol — one base pretrained on the pivot
language, cloned twice, each clone fine-tuned on one target direction, then
merged. Only the data changes. That is deliberate: the point of having several
is to ask whether the theta-vs-damage relationship is a property of BDH merging
or an artefact of one particular toy vocabulary, and that question is only
answerable if nothing else moves.

The model architecture is identical across all of them too (n=1024, d=64, L=4),
so merged neuron counts and parameter totals stay directly comparable. The one
exception is vocab_size, which is forced upward by the token layout when a
dataset has more concepts — see domains.layout().
"""
import dataclasses

from .bdh import TOY
from .domains import PHRASE_LEN, SEQ_LEN, layout


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    id: str
    label: str
    blurb: str          # one line, shown in the UI selector
    n_concepts: int
    phrase_len: int
    vocab_size: int

    @property
    def layout(self) -> dict:
        return layout(self.n_concepts)

    def config(self):
        """The BDH config for this dataset: TOY with vocab_size widened if the
        token layout needs it. Every other field is untouched, so n stays 1024
        and the merged axis stays 2048 across all datasets."""
        return dataclasses.replace(TOY, vocab_size=self.vocab_size)

    def check(self) -> None:
        lay = self.layout
        assert self.vocab_size >= lay["min_vocab"], (
            f"{self.id}: vocab_size {self.vocab_size} < {lay['min_vocab']} "
            f"needed for {self.n_concepts} concepts"
        )
        # longest sequence this dataset emits: BOS + phrase + SEP + phrase + EOS
        longest = 2 * self.phrase_len + 3
        assert longest <= SEQ_LEN, (
            f"{self.id}: sequences of {longest} tokens exceed SEQ_LEN {SEQ_LEN}"
        )


BASELINE = DatasetSpec(
    id="baseline",
    label="Baseline",
    blurb="24 concepts, 3-token phrases — the original scenario.",
    n_concepts=24,
    phrase_len=PHRASE_LEN,
    vocab_size=TOY.vocab_size,          # 96
)

LONG_PHRASE = DatasetSpec(
    id="long_phrase",
    label="Longer phrases",
    blurb="Same 24-concept vocabulary, but 6-token phrases — twice the context per example.",
    n_concepts=24,
    phrase_len=6,
    vocab_size=TOY.vocab_size,          # 96; layout is unchanged at 24 concepts
)

LARGE_VOCAB = DatasetSpec(
    id="large_vocab",
    label="Larger vocabulary",
    blurb="40 concepts instead of 24 — more to distinguish, same phrase length.",
    n_concepts=40,
    phrase_len=PHRASE_LEN,
    vocab_size=128,                     # layout needs 125; 96 would overflow
)

DATASETS = (BASELINE, LONG_PHRASE, LARGE_VOCAB)
BY_ID = {d.id: d for d in DATASETS}

for _d in DATASETS:
    _d.check()
