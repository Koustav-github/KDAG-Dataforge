"""Generates the one-page concept summary required by the problem statement
(PS p.13): a self-contained, authoritative briefing for an average data
scientist, 500-950 words, delivered as PDF.

This is deliberately a different register from docs/make_primer.py — that one
explains the project to a newcomer; this one is the judged deliverable and is
written for a reader who already knows what a Transformer is.

Run: uv run --with python-docx python docs/make_concept_summary.py
"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ACCENT = RGBColor(0x1F, 0x6F, 0x5C)
MUTED = RGBColor(0x5A, 0x57, 0x52)

doc = Document()
for s in doc.sections:
    s.top_margin = Inches(0.45)
    s.bottom_margin = Inches(0.45)
    s.left_margin = Inches(0.6)
    s.right_margin = Inches(0.6)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(9)
normal.paragraph_format.space_after = Pt(3.5)
normal.paragraph_format.line_spacing = 1.0


def para(runs, size=9, after=3.5, italic_all=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    for text, bold in runs:
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        if italic_all:
            r.italic = True
    return p


# ---- Title ------------------------------------------------------------------
t = doc.add_paragraph()
t.paragraph_format.space_after = Pt(0)
r = t.add_run("Composing parametric memory: what BDH's uniform neuron axis makes possible")
r.bold = True
r.font.size = Pt(13)
r.font.color.rgb = ACCENT

st = doc.add_paragraph()
st.paragraph_format.space_after = Pt(6)
r = st.add_run(
    "Concept: Parametric Memory in LLMs · Team Invariance, IIT Kharagpur · "
    "DataForge 2026, Pathway track · Artifact: bdh-merger.vercel.app"
)
r.font.size = Pt(8)
r.italic = True
r.font.color.rgb = MUTED

# ---- Body -------------------------------------------------------------------
para([
    ("The design pressure. ", True),
    ("A Transformer splits memory in two: fixed weights hold what it learned in training, a growing "
     "key–value cache holds what it just read. Neither composes well. Combining two fine-tuned "
     "Transformers means averaging their weights (model soups, arXiv:2203.05482) or adding task vectors "
     "(arXiv:2212.04089) — both force two specialists into one parameter set, and interference is the "
     "normal outcome, with a literature devoted to mitigating it (TIES-Merging, arXiv:2306.01708). The "
     "question underneath is where memory lives and whether trained capability can be ", False),
    ("composed", True),
    (" rather than retrained.", False),
])

para([
    ("What BDH changes. ", True),
    ("The Dragon Hatchling (BDH, arXiv:2509.26507) is a post-Transformer architecture that reformulates "
     "attention as synaptic memory: a population of n neurons whose pairwise connections strengthen "
     "Hebbianly as the model reads, so reasoning and short-term memory share one computational fabric "
     "instead of living in weights and a cache respectively. The structural consequence this submission "
     "exploits is that every learnable parameter hangs off one uniform neuron axis. In the reference "
     "implementation the model is five matrices — three of shape n×d (encoder, encoder_v, decoder) "
     "and two of shape Ω×d (embedding, lm_head, Ω = vocabulary size) — giving ", False),
    ("3nd + 2Ωd", True),
    (" parameters. Attention contributes no learnable parameters, and layers reuse the same matrices.", False),
])

para([
    ("The operation that unlocks. ", True),
    ("Because neurons are interchangeable units on a single axis, two BDH models can be ", False),
    ("concatenated", True),
    (": stack the three n×d matrices (n_A + n_B neurons) and average the two Ω×d matrices. This is "
     "the merge rule BDH's paper sketches in §7.1. It is neither weight averaging nor ensembling — no "
     "parameter is blended away, and inference is a single forward pass, not two. A Transformer admits no "
     "equivalent, because its parameters are not organised on one shared, permutation-tolerant axis.", False),
])

# ---- Comparison table -------------------------------------------------------
rows = [
    ("Approach", "Operation", "Params", "Inference", "Requires"),
    ("Ensembling", "run both, combine outputs", "2×", "2 passes", "nothing"),
    ("Weight averaging", "mean of weights", "1×", "1 pass", "shared init"),
    ("Task arithmetic", "add task vectors", "1×", "1 pass", "shared init"),
    ("BDH concatenation", "stack the neuron axis", "2×", "1 pass", "uniform neuron axis"),
]
table = doc.add_table(rows=len(rows), cols=5)
table.style = "Table Grid"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
widths = [Inches(1.15), Inches(2.05), Inches(0.5), Inches(0.72), Inches(1.35)]
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = table.cell(i, j)
        cell.width = widths[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(val)
        run.font.size = Pt(7.8)
        run.bold = (i == 0) or (j == 0 and i == 4)
doc.paragraphs[-1].paragraph_format.space_after = Pt(4)

cap = doc.add_paragraph()
cap.paragraph_format.space_before = Pt(2)
cap.paragraph_format.space_after = Pt(4)
r = cap.add_run(
    "Concatenation trades parameter count for keeping both specialists intact at single-pass inference "
    "cost — a trade only the uniform axis makes available."
)
r.font.size = Pt(7.8)
r.italic = True
r.font.color.rgb = MUTED

para([
    ("What we tested, and what failed. ", True),
    ("§7.1 raises composability as a conjecture — “when the model latent space promotes concept "
     "disentangling then it is feasible to directly compose concepts” — but does not test it. We did, "
     "on toy-scale reimplementations (n = 1024, ~209k parameters per parent) trained on paired synthetic "
     "languages sharing a pivot vocabulary, with θ controlling vocabulary overlap: 11 θ × 3 seeds "
     "× 3 datasets. Concatenation beat averaging in ", False),
    ("33 of 33 runs on all three datasets, at every θ", True),
    (", and merge damage tracks θ closely (corr = −0.991 over per-θ means). But ", False),
    ("our own hypothesis was falsified", True),
    (": we predicted damage would track measured representational overlap M (Hungarian-matched neuron "
     "activation correlation) and would localise to identifiable “collision” neurons. M stayed flat "
     "and its correlation with damage flipped sign under re-aggregation; ablating top-collision neurons "
     "never beat a size-matched random control, and was worse at k = 100. The disentanglement-implies-"
     "composability claim we were testing is the Transformer-side finding of arXiv:2305.12827; we could "
     "not confirm its BDH analogue.", False),
])

para([
    ("Evidence labels. ", True),
    ("Everything above is developer-reported by us, on a labelled toy reimplementation — not an official "
     "BDH model, not an independent reproduction, not a deployment. BDH's own §7.1 merge result is "
     "likewise reported by the architecture's authors. We are aware of no external reproduction of BDH "
     "merging. The artifact runs real forward passes in the browser (JS port verified against PyTorch to "
     "3.5e⁻⁷); precomputed sweeps are labelled as such in the UI.", False),
])

para([
    ("Roles of BDH and BDH-CQ. ", True),
    ("BDH is central: the merge operation is a direct consequence of its parameterisation. ", False),
    ("BDH-CQ has no direct role in this concept and we claim none", True),
    (" — it concerns in-context skill acquisition without weight updates, a different mechanism from "
     "composing already-trained parameters.", False),
])

para([
    ("The most important limitation. ", True),
    ("We cannot yet separate “representational overlap does not predict mergeability” from “our "
     "metric is mis-specified.” M pools over the whole neuron axis, while the divergence we suspect "
     "matters sits in the output pathway. Cross-dataset testing sharpened this rather than settling it: at "
     "40 concepts M stopped being flat and correlated with damage at −0.935, so M's behaviour is "
     "sensitive to vocabulary size rather than being a fixed property of the architecture. A parameter-count "
     "control — merged 2n against a single 2n parent, which would rule out a pure size effect — was "
     "planned and never run.", False),
])

para([
    ("Where to continue. ", True),
    ("Primary sources: BDH (arXiv:2509.26507, §7.1 for the merge rule); model soups (2203.05482); task "
     "arithmetic (2212.04089); TIES-Merging (2306.01708); weight disentanglement (2305.12827). Full "
     "numbers, the retracted claims, and every limitation are in the repository README.", False),
])

out_dir = Path(__file__).resolve().parent
out = out_dir / "concept-summary.docx"
doc.save(out)
print(f"wrote {out}")
