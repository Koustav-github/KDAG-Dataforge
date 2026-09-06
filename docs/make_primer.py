"""Generates docs/BDH_Model_Surgery_Primer.docx — the plain-language onboarding
explainer for the team. This is NOT the judge-facing one-page concept summary
(that has to be a technically dense briefing for a data scientist); this one is
deliberately written for someone who has never read the problem statement.

Run: uv run --with python-docx python docs/make_primer.py
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor, Inches

ACCENT = RGBColor(0x1F, 0x6F, 0x5C)
MUTED = RGBColor(0x60, 0x5D, 0x57)

doc = Document()

# Tight margins so this genuinely lands on one page.
for s in doc.sections:
    s.top_margin = Inches(0.5)
    s.bottom_margin = Inches(0.5)
    s.left_margin = Inches(0.62)
    s.right_margin = Inches(0.62)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(9.5)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.0


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.color.rgb = ACCENT
    return p


def body(runs):
    """runs: list of (text, bold) tuples."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
    return p


def bullet(runs):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.22)
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(9.5)
    return p


# ---- Title ------------------------------------------------------------------
t = doc.add_paragraph()
t.paragraph_format.space_after = Pt(1)
r = t.add_run("Model Surgery: a plain-language primer")
r.bold = True
r.font.size = Pt(15)
r.font.color.rgb = ACCENT

sub = doc.add_paragraph()
sub.paragraph_format.space_after = Pt(7)
r = sub.add_run(
    "What BDH is, how merging two models works, and why any of it is new. "
    "Written for someone who has not read the problem statement. Team Invariance, IIT Kharagpur."
)
r.font.size = Pt(9)
r.font.color.rgb = MUTED
r.italic = True

# ---- 1 ----------------------------------------------------------------------
heading("1. The problem with how today's models remember")
body([
    ("A Transformer — the architecture behind essentially every chatbot you have used — reads text by "
     "comparing every word to every other word. To answer about word 500 it keeps all 499 earlier words "
     "in a scratchpad called the ", False),
    ("KV cache", True),
    (". That scratchpad grows with every token. Longer conversation, more memory, slower replies. Memory "
     "and reasoning also live in different places: the weights hold what the model learned in training, "
     "the cache holds what it read a second ago, and the two never really become one thing.", False),
])

# ---- 2 ----------------------------------------------------------------------
heading("2. What BDH does differently")
body([
    ("Dragon Hatchling (BDH, Pathway, arXiv:2509.26507) is a ", False),
    ("post-Transformer", True),
    (" architecture that borrows its shape from a brain instead. Picture a population of ", False),
    ("neurons", True),
    (" wired to each other by ", False),
    ("synapses", True),
    (". When the model reads, connections between co-active neurons briefly strengthen — a Hebbian "
     "\"fire together, wire together\" update. That strengthening ", False),
    ("is", True),
    (" the model's short-term memory. Attention is not a separate lookup over a growing cache; it is the "
     "current state of the wiring. Memory and reasoning share one fabric.", False),
])
body([
    ("The consequence that matters for this project is structural. In BDH every parameter hangs off one "
     "uniform axis — the neuron count ", False),
    ("n", True),
    (". The parameter count is literally 3nd + 2Ωd. There is no stack of differently-shaped blocks that "
     "must line up in a particular order; there is a population of interchangeable units.", False),
])

# ---- 3 ----------------------------------------------------------------------
heading("3. Why that makes “model surgery” possible")
body([
    ("Say you train two models: one translates a shared concept language into Language A, the other into "
     "Language B. You now want one model that does both. Your options in a Transformer are awkward:", False),
])
bullet([
    ("Average the weights", True),
    (" (“model soups”, arXiv:2203.05482). Two specialists become one compromise — like averaging "
     "two recipes and getting mush.", False),
])
bullet([
    ("Add task vectors", True),
    (" (arXiv:2212.04089). Better, but still one set of neurons carrying both jobs.", False),
])
body([
    ("BDH allows a third move that Transformers simply cannot express: ", False),
    ("concatenation", True),
    (". Because neurons are interchangeable units on one axis, you can lay the two populations side by "
     "side — model A's 1024 neurons and model B's 1024 neurons become one 2048-neuron model. Nobody is "
     "averaged away. It is less like blending two recipes and more like hiring both chefs. This is the "
     "merge rule the BDH paper proposes in §7.1: concatenate along the neuron axis, average the rest.", False),
])

# ---- 4 ----------------------------------------------------------------------
heading("4. What we built, and what actually happened")
body([
    ("We trained pairs of small BDH models (n = 1024, ~209k parameters each) on two synthetic languages "
     "sharing a pivot vocabulary, with a dial θ controlling how much vocabulary they share. Then we merged "
     "them and measured the damage. The artifact lets you move θ yourself and watch it happen live in the "
     "browser — real forward passes, not an animation.", False),
])
bullet([
    ("Concatenation beat averaging in 33 of 33 runs, on all three datasets, at every θ.", True),
    (" This is the architectural point, and it replicated everywhere.", False),
])
bullet([
    ("Damage is strongly predictable from θ ", False),
    ("(corr = −0.991)", True),
    (" — the more vocabulary two models share, the better they fuse.", False),
])
bullet([
    ("Our original hypothesis was falsified.", True),
    (" We set out to show damage is predictable from measured neuron overlap, and that it localises to "
     "identifiable “collision” neurons. Neither survived our own data. We report that rather than quietly "
     "changing the question.", False),
])

# ---- 5 ----------------------------------------------------------------------
heading("5. How to appreciate why this is novel")
body([
    ("Two things, and they are different.", False),
])
bullet([
    ("The operation is architecture-specific.", True),
    (" “Combine two models” normally means ensembling (run both, vote) or averaging (blend the weights). "
     "Concatenation is a genuinely third thing, and it is available ", False),
    ("because", True),
    (" of how BDH is built. It is a concrete demonstration that architecture choices unlock operations, "
     "not just efficiency.", False),
])
bullet([
    ("The honesty is the contribution.", True),
    (" The BDH paper poses composability as an open question and does not test it. We ran the experiment, "
     "got a negative result on our own hypothesis, and published the negative result with the retraction "
     "visible. A demo that only confirms what its authors hoped teaches nothing about how research "
     "actually goes.", False),
])

foot = doc.add_paragraph()
foot.paragraph_format.space_before = Pt(6)
r = foot.add_run(
    "Caveat, stated plainly: our models are toy-scale reimplementations (n = 1024 vs the paper's 24,576) "
    "trained on synthetic data. They are labelled as such throughout and are not official BDH models. "
    "Full numbers, limitations and sources are in the repository README."
)
r.font.size = Pt(8)
r.italic = True
r.font.color.rgb = MUTED

out = Path(__file__).resolve().parent / "BDH_Model_Surgery_Primer.docx"
doc.save(out)
print(f"wrote {out}")
