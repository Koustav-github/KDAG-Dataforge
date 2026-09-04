# Model Surgery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how BDH merge damage varies with representational overlap between two independently fine-tuned models, and ship an interactive browser explainer built on that result.

**Architecture:** Python trains toy BDH-GPU models with the fast parallel attention form, sweeps 11 overlap levels × 3 seeds, and exports weights plus sweep metrics as static data. The browser re-implements the mathematically equivalent recurrent form (verified to 3e-07) so merges, generation and neuron ablation all run live client-side with no ML runtime.

**Tech Stack:** PyTorch (CPU), NumPy, SciPy, pytest, uv · Vite + Svelte, plain-JS forward pass over Float32Array, Canvas 2D, hand-rolled SVG

**Spec:** `docs/Model_Surgery_Spec.docx`

## Global Constraints

- Python `>=3.12`, managed by uv. Model package lives in `model/`, importable as `bdh_surgery`.
- Model deps limited to: `torch`, `numpy`, `scipy`. Dev: `pytest`. No Hydra, W&B, MLflow, Lightning.
- Web deps limited to: `vite`, `svelte`. No ONNX Runtime Web, TensorFlow.js, GSAP, Framer Motion, D3 (full), Chart.js.
- `model/src/bdh_surgery/bdh.py` is adapted from `github.com/pathwaycom/bdh` (MIT). Keep the copyright header and record it in `PROVENANCE.md`.
- Toy config, fixed everywhere: `n_layer=4, n_embd=64, n_head=1, mlp_internal_dim_multiplier=16, vocab_size=96, dropout=0.0` → neuron axis `N=1024`, **208,896 params** per parent.
- Merged model: multiplier doubled to `32` → `N=2048`, **405,504 params**.
- Merge rule (arXiv:2509.26507 §7.1), exact tensor mapping:
  - **Concatenate** — `decoder` dim 0, `encoder` dim 2, `encoder_v` dim 2, `attn.freqs` dim 3
  - **Average** — `lm_head`, `embed.weight`
  - `ln` is `LayerNorm(elementwise_affine=False, bias=False)` — zero parameters, nothing to merge
- Sweep: θ ∈ {0.0, 0.1, …, 1.0} (11 points) × 3 seeds. Seeds are non-negotiable; every plotted point carries an error bar.
- All artifacts exported to `web/public/data/` are **committed to git**.
- Every UI element labelled `live` / `precomputed` / `illustration`. Every cap stated on screen.

---

## File Structure

| File | Responsibility |
|---|---|
| `model/src/bdh_surgery/bdh.py` | BDH-GPU model, parallel form. Adapted from pathwaycom/bdh (MIT). |
| `model/src/bdh_surgery/recurrent.py` | Equivalent recurrent (Eq. 8) form. Reference for the JS port. |
| `model/src/bdh_surgery/domains.py` | Concept set, θ-parameterised target languages, corpus generation. |
| `model/src/bdh_surgery/train.py` | Base pretraining and per-direction fine-tuning. |
| `model/src/bdh_surgery/merge.py` | `merge_concat`, `merge_average`, tensor classification. |
| `model/src/bdh_surgery/overlap.py` | Probe → activation profiles → Hungarian matching → M. |
| `model/src/bdh_surgery/damage.py` | Held-out loss deltas → D, per direction. |
| `model/src/bdh_surgery/ablate.py` | Collision-neuron identification, masking, random baseline. |
| `model/src/bdh_surgery/sweep.py` | Orchestrates 33 pairs; writes `runs.csv`. |
| `model/src/bdh_surgery/export.py` | Writes `manifest.json`, `*.bin`, `sweep.json`, `probes.json`. |
| `web/src/bdh_forward.js` | Recurrent forward pass over Float32Array. |
| `web/src/merge.js` | Client-side concat/average of loaded parent weights. |
| `web/src/lib/*.svelte` | UI components, one per act. |

---

### Task 1: Toy BDH model

**Files:**
- Create: `model/src/bdh_surgery/bdh.py`
- Test: `model/tests/test_bdh.py`

**Interfaces:**
- Produces: `BDHConfig` dataclass (fields `n_layer, n_embd, dropout, n_head, mlp_internal_dim_multiplier, vocab_size`); `BDH(config)` with `forward(idx, targets=None) -> (logits, loss)`; module-level `TOY = BDHConfig(n_layer=4, n_embd=64, n_head=1, mlp_internal_dim_multiplier=16, vocab_size=96, dropout=0.0)`; helper `neuron_axis(cfg) -> int` returning `cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_bdh.py
import torch
from bdh_surgery.bdh import BDH, BDHConfig, TOY, neuron_axis


def test_toy_param_count_matches_3nd_plus_2vd():
    net = BDH(TOY)
    n, d, v = neuron_axis(TOY), TOY.n_embd, TOY.vocab_size
    assert neuron_axis(TOY) == 1024
    assert sum(p.numel() for p in net.parameters()) == 3 * n * d + 2 * v * d == 208896


def test_layernorm_is_parameter_free():
    assert sum(p.numel() for p in BDH(TOY).ln.parameters()) == 0


def test_forward_shapes_and_loss():
    net = BDH(TOY)
    idx = torch.randint(0, TOY.vocab_size, (2, 16))
    logits, loss = net(idx, targets=idx)
    assert logits.shape == (2, 16, TOY.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_bdh.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.bdh'` (or ImportError on `TOY`).

- [ ] **Step 3: Write the implementation**

Copy `bdh.py` from `github.com/pathwaycom/bdh` verbatim, preserving its `# Copyright 2025 Pathway Technology, Inc.` header. Then append:

```python
TOY = BDHConfig(
    n_layer=4, n_embd=64, n_head=1,
    mlp_internal_dim_multiplier=16, vocab_size=96, dropout=0.0,
)


def neuron_axis(cfg: BDHConfig) -> int:
    """Size of the free neuron dimension n that merging concatenates along."""
    return cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_bdh.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/bdh.py model/tests/test_bdh.py
git commit -m "feat(model): toy BDH-GPU config, adapted from pathwaycom/bdh (MIT)"
```

---

### Task 2: Recurrent inference form

The browser cannot afford the parallel form's O(T²N) cost per generated token. This task builds the O(N·D)-per-token recurrent form and proves it equivalent — it is the reference the JS port is checked against.

**Files:**
- Create: `model/src/bdh_surgery/recurrent.py`
- Test: `model/tests/test_recurrent.py`

**Interfaces:**
- Consumes: `BDH`, `BDHConfig`, `neuron_axis` from Task 1.
- Produces: `recurrent_logits(net: BDH, idx: torch.Tensor) -> torch.Tensor` of shape `(1, T, vocab_size)`; requires `n_head == 1` and batch size 1.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_recurrent.py
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.recurrent import recurrent_logits


def test_recurrent_matches_parallel():
    torch.manual_seed(0)
    net = BDH(TOY).eval()
    idx = torch.randint(0, TOY.vocab_size, (1, 24))
    with torch.no_grad():
        parallel, _ = net(idx)
    recurrent = recurrent_logits(net, idx)
    assert recurrent.shape == parallel.shape
    assert (parallel - recurrent).abs().max().item() < 1e-4
    assert torch.equal(parallel.argmax(-1), recurrent.argmax(-1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_recurrent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.recurrent'`.

- [ ] **Step 3: Write the implementation**

State `rho` of shape `(N, D)` per layer accumulates `QRᵀ ⊗ x`. Read happens strictly before write, matching the reference's `tril(diagonal=-1)` (τ < t).

```python
# model/src/bdh_surgery/recurrent.py
import torch
import torch.nn.functional as F

from .bdh import BDH, Attention, neuron_axis


@torch.no_grad()
def recurrent_logits(net: BDH, idx: torch.Tensor) -> torch.Tensor:
    """Eq. (8) recurrent form. Equivalent to net(idx)[0]; O(N*D) state per token."""
    cfg = net.config
    assert cfg.n_head == 1, "recurrent form implemented for n_head=1 only"
    assert idx.shape[0] == 1, "recurrent form is batch-size-1 only"
    n, d, T = neuron_axis(cfg), cfg.n_embd, idx.shape[1]
    freqs = net.attn.freqs.view(n)
    rho = [torch.zeros(n, d) for _ in range(cfg.n_layer)]
    outs = []
    for t in range(T):
        x = net.ln(net.embed(idx[:, t : t + 1]).unsqueeze(1))          # (1,1,1,D)
        for layer in range(cfg.n_layer):
            x_sparse = F.relu(x @ net.encoder)                          # (1,1,1,N)
            phases = (t * freqs).view(1, 1, 1, n)
            qr = Attention.rope(phases, x_sparse).view(n)
            y_kv = (qr @ rho[layer]).view(1, 1, 1, d)                   # read before write
            rho[layer] = rho[layer] + torch.outer(qr, x.view(d))
            y_sparse = F.relu(net.ln(y_kv) @ net.encoder_v)
            y_mlp = (x_sparse * y_sparse).view(1, 1, 1, n) @ net.decoder
            x = net.ln(x + net.ln(y_mlp))
        outs.append(x.view(d))
    return torch.stack(outs).unsqueeze(0) @ net.lm_head
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_recurrent.py -v`
Expected: PASS. Observed max abs diff during spike: 2.7e-07.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/recurrent.py model/tests/test_recurrent.py
git commit -m "feat(model): recurrent Eq.8 form with parity test against parallel form"
```

---

### Task 3: Synthetic domains with the overlap knob

**Files:**
- Create: `model/src/bdh_surgery/domains.py`
- Test: `model/tests/test_domains.py`

**Design.** `K=24` concepts. Pivot language P uses token ids `1..24`. Targets A and B draw surface tokens from ids `25..72`. For a θ fraction of concepts, A and B assign the **same** surface token; for the rest they assign distinct ones. Sequences are `[BOS_A|BOS_B] p1 p2 p3 SEP t1 t2 t3 EOS`. Special ids: `0=PAD, 73=BOS_A, 74=BOS_B, 75=SEP, 76=EOS`. `vocab_size=96` leaves headroom.

**Interfaces:**
- Produces: `Vocab` constants `PAD=0, BOS_A=73, BOS_B=74, SEP=75, EOS=76`, `N_CONCEPTS=24`, `SEQ_LEN=16`; `build_lexicons(theta: float, seed: int) -> tuple[list[int], list[int]]` returning `(lex_a, lex_b)` each length 24; `make_corpus(lex, bos, n_seqs, seed, phrase_len=3) -> torch.Tensor` of shape `(n_seqs, SEQ_LEN)` int64; `make_pivot_corpus(n_seqs, seed) -> torch.Tensor` for base pretraining; `measured_lexicon_overlap(lex_a, lex_b) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_domains.py
import torch
from bdh_surgery.domains import (
    N_CONCEPTS, SEQ_LEN, BOS_A, BOS_B, SEP, EOS,
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
    assert corpus.max().item() <= N_CONCEPTS or (corpus >= 73).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_domains.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.domains'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/domains.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_domains.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/domains.py model/tests/test_domains.py
git commit -m "feat(model): synthetic pivot/target domains with overlap knob theta"
```

---

### Task 4: Base pretraining and fine-tuning

Mirrors §7.1's protocol exactly: train one base, clone it twice, fine-tune each clone on one direction. The shared ancestor is what makes averaging `embed`/`lm_head` meaningful. The base is θ-independent, so **one base per seed is reused across all 11 θ values** — 3 bases, 66 fine-tunes.

**Files:**
- Create: `model/src/bdh_surgery/train.py`
- Test: `model/tests/test_train.py`

**Interfaces:**
- Consumes: Task 1 `BDH/TOY`, Task 3 corpus builders.
- Produces: `train_model(net, corpus, steps, lr, seed, batch_size=32) -> list[float]` (loss history, mutates `net`); `train_base(seed, steps=400) -> BDH`; `finetune(base, lex, bos, seed, steps=600) -> BDH` (clones internally, never mutates `base`); `evaluate(net, corpus) -> float` mean held-out cross-entropy.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_train.py
import copy
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.domains import BOS_A, build_lexicons, make_corpus, make_pivot_corpus
from bdh_surgery.train import train_model, train_base, finetune, evaluate


def test_training_reduces_loss():
    torch.manual_seed(0)
    net = BDH(TOY)
    corpus = make_pivot_corpus(256, seed=1)
    history = train_model(net, corpus, steps=60, lr=1e-3, seed=0)
    assert len(history) == 60
    assert sum(history[-5:]) / 5 < sum(history[:5]) / 5


def test_finetune_does_not_mutate_base():
    base = train_base(seed=0, steps=20)
    before = copy.deepcopy(base.state_dict())
    lex_a, _ = build_lexicons(0.5, seed=0)
    finetune(base, lex_a, BOS_A, seed=0, steps=20)
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())


def test_evaluate_returns_positive_scalar():
    net = BDH(TOY)
    assert evaluate(net, make_pivot_corpus(64, seed=2)) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.train'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/train.py
import copy

import torch

from .bdh import BDH, TOY
from .domains import make_corpus, make_pivot_corpus

TRAIN_SEQS = 2048
EVAL_SEQS = 256


def train_model(net: BDH, corpus: torch.Tensor, steps: int, lr: float,
                seed: int, batch_size: int = 32) -> list[float]:
    """AdamW with linear decay, following arXiv:2509.26507 Appendix B.4 in spirit."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=steps)
    net.train()
    history = []
    for _ in range(steps):
        pick = torch.randint(0, corpus.shape[0], (batch_size,), generator=g)
        batch = corpus[pick]
        _, loss = net(batch[:, :-1], targets=batch[:, 1:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        history.append(loss.item())
    net.eval()
    return history


@torch.no_grad()
def evaluate(net: BDH, corpus: torch.Tensor) -> float:
    net.eval()
    _, loss = net(corpus[:, :-1], targets=corpus[:, 1:])
    return loss.item()


def train_base(seed: int, steps: int = 400) -> BDH:
    """Theta-independent pivot-language base. Cloned by both parents."""
    torch.manual_seed(seed)
    net = BDH(TOY)
    train_model(net, make_pivot_corpus(TRAIN_SEQS, seed=seed), steps, 1e-3, seed)
    return net


def finetune(base: BDH, lex: list[int], bos: int, seed: int, steps: int = 600) -> BDH:
    """Clone the base and fine-tune on one translation direction."""
    net = copy.deepcopy(base)
    train_model(net, make_corpus(lex, bos, TRAIN_SEQS, seed=seed), steps, 5e-4, seed)
    return net
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_train.py -v`
Expected: 3 passed. If `test_training_reduces_loss` is flaky, raise `steps` to 100 — do **not** loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/train.py model/tests/test_train.py
git commit -m "feat(model): base pretraining and per-direction finetuning"
```

---

### Task 5: The merge

The live-defence exhibit. The tensor classification must be derived from shape, never hardcoded by name, so it cannot silently drift.

**Files:**
- Create: `model/src/bdh_surgery/merge.py`
- Test: `model/tests/test_merge.py` (replace the Task-0 skeleton)

**Interfaces:**
- Consumes: Task 1 `BDH/BDHConfig/TOY/neuron_axis`.
- Produces: `CONCAT_TENSORS: dict[str, int]` mapping tensor name → concat axis; `AVERAGE_TENSORS: tuple[str, ...]`; `merged_config(cfg) -> BDHConfig`; `merge_concat(a: BDH, b: BDH) -> BDH`; `merge_average(a: BDH, b: BDH) -> BDH`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_merge.py
import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.merge import (
    CONCAT_TENSORS, AVERAGE_TENSORS, merged_config, merge_concat, merge_average,
)


def test_every_parameter_is_classified_exactly_once():
    names = {k for k, _ in BDH(TOY).named_parameters()}
    classified = set(CONCAT_TENSORS) | set(AVERAGE_TENSORS)
    assert names - {"attn.freqs"} <= classified
    assert set(CONCAT_TENSORS).isdisjoint(AVERAGE_TENSORS)


def test_concat_axis_is_the_neuron_axis_for_every_concat_tensor():
    net, n = BDH(TOY), neuron_axis(TOY)
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    for name, axis in CONCAT_TENSORS.items():
        assert tensors[name].shape[axis] == n, f"{name} axis {axis} is not the n axis"


def test_average_tensors_have_no_neuron_axis():
    net, n = BDH(TOY), neuron_axis(TOY)
    tensors = dict(net.named_parameters())
    for name in AVERAGE_TENSORS:
        assert n not in tuple(tensors[name].shape)


def test_merge_concat_doubles_n_and_matches_param_formula():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_concat(a, b)
    assert neuron_axis(m.config) == 2 * neuron_axis(TOY) == 2048
    assert sum(p.numel() for p in m.parameters()) == 3 * 2048 * 64 + 2 * 96 * 64 == 405504


def test_merge_concat_places_parents_side_by_side():
    a, b = BDH(TOY), BDH(TOY)
    m, n = merge_concat(a, b), neuron_axis(TOY)
    assert torch.equal(m.decoder[:n], a.decoder)
    assert torch.equal(m.decoder[n:], b.decoder)
    assert torch.equal(m.encoder[..., :n], a.encoder)
    assert torch.equal(m.encoder[..., n:], b.encoder)


def test_merge_averages_the_non_neuron_tensors():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_concat(a, b)
    assert torch.allclose(m.lm_head, (a.lm_head + b.lm_head) / 2)
    assert torch.allclose(m.embed.weight, (a.embed.weight + b.embed.weight) / 2)


def test_merge_average_keeps_size_and_blends_all_tensors():
    a, b = BDH(TOY), BDH(TOY)
    m = merge_average(a, b)
    assert neuron_axis(m.config) == neuron_axis(TOY)
    assert torch.allclose(m.decoder, (a.decoder + b.decoder) / 2)


def test_both_merges_run_a_forward_pass():
    a, b = BDH(TOY), BDH(TOY)
    idx = torch.randint(0, TOY.vocab_size, (1, 12))
    for m in (merge_concat(a, b), merge_average(a, b)):
        logits, _ = m(idx)
        assert logits.shape == (1, 12, TOY.vocab_size)
        assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'CONCAT_TENSORS'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/merge.py
"""Merge rule from arXiv:2509.26507 §7.1.

Concatenate every tensor carrying the neuron dimension n along that axis;
average everything else. LayerNorm is parameter-free, so it needs no handling.
The RoPE frequency buffer is concatenated rather than recomputed for the new n,
exactly as the paper prescribes.
"""
import copy
import dataclasses

import torch

from .bdh import BDH, BDHConfig, neuron_axis

CONCAT_TENSORS: dict[str, int] = {
    "decoder": 0,       # (n, D)
    "encoder": 2,       # (nh, D, n)
    "encoder_v": 2,     # (nh, D, n)
    "attn.freqs": 3,    # (1, 1, 1, n) — buffer, not a parameter
}
AVERAGE_TENSORS: tuple[str, ...] = ("lm_head", "embed.weight")


def merged_config(cfg: BDHConfig) -> BDHConfig:
    return dataclasses.replace(
        cfg, mlp_internal_dim_multiplier=cfg.mlp_internal_dim_multiplier * 2)


def _blend(out: BDH, a: BDH, b: BDH, concat: bool) -> BDH:
    ta = dict(a.named_parameters()) | {"attn.freqs": a.attn.freqs}
    tb = dict(b.named_parameters()) | {"attn.freqs": b.attn.freqs}
    to = dict(out.named_parameters()) | {"attn.freqs": out.attn.freqs}
    with torch.no_grad():
        for name, axis in CONCAT_TENSORS.items():
            if concat:
                to[name].copy_(torch.cat([ta[name], tb[name]], dim=axis))
            else:
                to[name].copy_((ta[name] + tb[name]) / 2)
        for name in AVERAGE_TENSORS:
            to[name].copy_((ta[name] + tb[name]) / 2)
    return out.eval()


def merge_concat(a: BDH, b: BDH) -> BDH:
    """Grow along n: the merged model has n_a + n_b neurons, both parents intact."""
    assert a.config == b.config, "parents must share a config"
    return _blend(BDH(merged_config(a.config)), a, b, concat=True)


def merge_average(a: BDH, b: BDH) -> BDH:
    """The Transformer-style control: same size, representations blended."""
    assert a.config == b.config, "parents must share a config"
    return _blend(BDH(copy.deepcopy(a.config)), a, b, concat=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_merge.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/merge.py model/tests/test_merge.py
git commit -m "feat(model): concat and average merges with shape-derived tensor classification"
```

---

### Task 6: Measured representational overlap (M)

**Files:**
- Create: `model/src/bdh_surgery/overlap.py`
- Test: `model/tests/test_overlap.py`

**Interfaces:**
- Consumes: Task 1 `BDH`, Task 3 `make_pivot_corpus`.
- Produces: `activation_profiles(net, probe) -> np.ndarray` shape `(n, n_probe_tokens)`; `matched_correlations(net_a, net_b, probe) -> np.ndarray` shape `(n,)` sorted descending; `overlap_at(matched, tau) -> float`; `mean_matched_correlation(matched) -> float`; `TAUS = (0.5, 0.7, 0.9)`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_overlap.py
import copy
import numpy as np
import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.domains import make_pivot_corpus
from bdh_surgery.overlap import (
    TAUS, activation_profiles, matched_correlations,
    overlap_at, mean_matched_correlation,
)


def test_profiles_have_one_row_per_neuron():
    net, probe = BDH(TOY).eval(), make_pivot_corpus(8, seed=1)
    prof = activation_profiles(net, probe)
    assert prof.shape[0] == neuron_axis(TOY)
    assert prof.shape[1] == probe.shape[0] * probe.shape[1]


def test_identical_models_have_overlap_one():
    torch.manual_seed(0)
    net, probe = BDH(TOY).eval(), make_pivot_corpus(8, seed=1)
    matched = matched_correlations(net, copy.deepcopy(net), probe)
    assert matched.shape == (neuron_axis(TOY),)
    assert overlap_at(matched, 0.9) > 0.99
    assert mean_matched_correlation(matched) > 0.99


def test_independent_models_have_lower_overlap_than_identical_ones():
    torch.manual_seed(0); a = BDH(TOY).eval()
    torch.manual_seed(999); b = BDH(TOY).eval()
    probe = make_pivot_corpus(8, seed=1)
    assert (mean_matched_correlation(matched_correlations(a, b, probe))
            < mean_matched_correlation(matched_correlations(a, copy.deepcopy(a), probe)))


def test_overlap_at_is_monotone_in_tau():
    torch.manual_seed(0); a = BDH(TOY).eval()
    torch.manual_seed(7); b = BDH(TOY).eval()
    matched = matched_correlations(a, b, make_pivot_corpus(8, seed=1))
    vals = [overlap_at(matched, t) for t in TAUS]
    assert vals == sorted(vals, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in vals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_overlap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.overlap'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/overlap.py
"""Representational overlap M between two trained parents.

BDH neurons are sparse and reported as near-monosemantic (§6.2-6.4), so a
matching-based measure is the natural choice over a subspace measure like CKA:
we ask which of A's neurons have a counterpart in B, not whether the two span
a similar space.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .bdh import BDH, neuron_axis

TAUS = (0.5, 0.7, 0.9)


@torch.no_grad()
def activation_profiles(net: BDH, probe: torch.Tensor) -> np.ndarray:
    """Each neuron's sparse activation y over every probe token. Shape (n, tokens)."""
    net.eval()
    x = net.ln(net.embed(probe).unsqueeze(1))
    acts = []
    for _ in range(net.config.n_layer):
        x_sparse = F.relu(x @ net.encoder)
        y_kv = net.ln(net.attn(Q=x_sparse, K=x_sparse, V=x))
        y_sparse = F.relu(y_kv @ net.encoder_v)
        xy = x_sparse * y_sparse
        acts.append(xy)
        y = net.ln(xy.transpose(1, 2).reshape(x.shape[0], 1, probe.shape[1], -1)
                   @ net.decoder)
        x = net.ln(x + y)
    stacked = torch.stack(acts).mean(0)                       # average over layers
    n = neuron_axis(net.config)
    return stacked.reshape(-1, n).T.cpu().numpy()             # (n, tokens)


def _zscore(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(axis=1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def matched_correlations(net_a: BDH, net_b: BDH, probe: torch.Tensor) -> np.ndarray:
    """Hungarian-matched per-neuron correlations, sorted descending."""
    a, b = _zscore(activation_profiles(net_a, probe)), _zscore(activation_profiles(net_b, probe))
    corr = a @ b.T                                            # (n, n) cosine similarity
    rows, cols = linear_sum_assignment(-corr)                 # maximise total correlation
    return np.sort(corr[rows, cols])[::-1]


def overlap_at(matched: np.ndarray, tau: float) -> float:
    """M = fraction of matched neuron pairs correlating above tau."""
    return float((matched >= tau).mean())


def mean_matched_correlation(matched: np.ndarray) -> float:
    """Thresholdless fallback, used if matched counting proves noisy at n=1024."""
    return float(matched.mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_overlap.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/overlap.py model/tests/test_overlap.py
git commit -m "feat(model): Hungarian-matched representational overlap metric"
```

---

### Task 7: Merge damage (D)

**Files:**
- Create: `model/src/bdh_surgery/damage.py`
- Test: `model/tests/test_damage.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4.
- Produces: `@dataclass Damage` with fields `d_a: float, d_b: float, d_mean: float, loss_merged_a: float, loss_merged_b: float, loss_parent_a: float, loss_parent_b: float`; `measure_damage(merged, parent_a, parent_b, lex_a, lex_b, seed) -> Damage`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_damage.py
import copy
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.domains import BOS_A, BOS_B, build_lexicons
from bdh_surgery.damage import Damage, measure_damage


def test_self_merge_of_identical_parents_reports_finite_damage():
    torch.manual_seed(0)
    p = BDH(TOY).eval()
    lex_a, lex_b = build_lexicons(0.5, seed=0)
    d = measure_damage(copy.deepcopy(p), p, copy.deepcopy(p), lex_a, lex_b, seed=1)
    assert isinstance(d, Damage)
    assert all(map(torch.isfinite, map(torch.tensor, (d.d_a, d.d_b, d.d_mean))))
    assert abs(d.d_mean - (d.d_a + d.d_b) / 2) < 1e-9


def test_damage_is_zero_when_merged_equals_parents():
    torch.manual_seed(0)
    p = BDH(TOY).eval()
    lex_a, lex_b = build_lexicons(1.0, seed=0)
    d = measure_damage(copy.deepcopy(p), p, copy.deepcopy(p), lex_a, lex_b, seed=1)
    assert abs(d.d_a) < 1e-6 and abs(d.d_b) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_damage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.damage'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/damage.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_damage.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/damage.py model/tests/test_damage.py
git commit -m "feat(model): per-direction merge damage metric"
```

---

### Task 8: Collision neurons, ablation, and the random baseline

**Files:**
- Create: `model/src/bdh_surgery/ablate.py`
- Test: `model/tests/test_ablate.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 6.
- Produces: `collision_scores(merged, probe_a, probe_b) -> np.ndarray` shape `(2n,)`; `top_collision_neurons(scores, k) -> np.ndarray` of indices; `random_neurons(n_total, k, seed) -> np.ndarray`; `ablate(merged, indices) -> BDH` (returns a copy, never mutates).

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_ablate.py
import numpy as np
import torch
from bdh_surgery.bdh import BDH, TOY, neuron_axis
from bdh_surgery.domains import BOS_A, BOS_B, build_lexicons, make_corpus
from bdh_surgery.merge import merge_concat
from bdh_surgery.ablate import (
    collision_scores, top_collision_neurons, random_neurons, ablate,
)


def _fixture():
    torch.manual_seed(0)
    m = merge_concat(BDH(TOY).eval(), BDH(TOY).eval())
    lex_a, lex_b = build_lexicons(0.5, seed=0)
    return (m, make_corpus(lex_a, BOS_A, 8, seed=1), make_corpus(lex_b, BOS_B, 8, seed=2))


def test_collision_scores_cover_the_merged_neuron_axis():
    m, pa, pb = _fixture()
    s = collision_scores(m, pa, pb)
    assert s.shape == (neuron_axis(m.config),) == (2048,)
    assert np.isfinite(s).all()


def test_top_collision_returns_k_unique_indices():
    m, pa, pb = _fixture()
    idx = top_collision_neurons(collision_scores(m, pa, pb), k=50)
    assert idx.shape == (50,) and len(set(idx.tolist())) == 50


def test_random_baseline_is_disjoint_size_matched_and_reproducible():
    a = random_neurons(2048, 50, seed=3)
    assert a.shape == (50,) and len(set(a.tolist())) == 50
    assert np.array_equal(a, random_neurons(2048, 50, seed=3))
    assert not np.array_equal(a, random_neurons(2048, 50, seed=4))


def test_ablate_zeroes_only_the_named_neurons_and_leaves_merged_untouched():
    m, pa, pb = _fixture()
    before = m.decoder.clone()
    idx = np.array([0, 5, 9])
    out = ablate(m, idx)
    assert torch.equal(m.decoder, before), "ablate must not mutate its input"
    assert out.decoder[idx].abs().sum().item() == 0
    keep = np.setdiff1d(np.arange(neuron_axis(m.config)), idx)
    assert torch.equal(out.decoder[keep], before[keep])


def test_ablated_model_still_runs():
    m, pa, pb = _fixture()
    out = ablate(m, top_collision_neurons(collision_scores(m, pa, pb), k=20))
    logits, _ = out(torch.randint(0, TOY.vocab_size, (1, 12)))
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_ablate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.ablate'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/ablate.py
"""Collision neurons: units in the merged model that respond to both target
languages. A neuron inherited cleanly from one parent should be selective for
that parent's direction; one that fires for both is where the two models
compete for the same output channel.
"""
import copy

import numpy as np
import torch

from .bdh import BDH
from .overlap import activation_profiles


def collision_scores(merged: BDH, probe_a: torch.Tensor,
                     probe_b: torch.Tensor) -> np.ndarray:
    """High where a neuron is strongly and equally active on both directions."""
    a = activation_profiles(merged, probe_a).mean(axis=1)
    b = activation_profiles(merged, probe_b).mean(axis=1)
    joint = np.minimum(a, b)                     # active on both
    imbalance = np.abs(a - b) / (a + b + 1e-8)   # 0 when equally active
    return joint * (1.0 - imbalance)


def top_collision_neurons(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(scores)[::-1][:k].copy()


def random_neurons(n_total: int, k: int, seed: int) -> np.ndarray:
    """Size-matched control. Without this, 'we ablated and it improved' proves nothing."""
    return np.random.default_rng(seed).choice(n_total, size=k, replace=False)


def ablate(merged: BDH, indices: np.ndarray) -> BDH:
    """Zero the named neurons everywhere they appear on the n axis. Returns a copy."""
    out = copy.deepcopy(merged).eval()
    idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    with torch.no_grad():
        out.decoder[idx] = 0.0          # (n, D)
        out.encoder[..., idx] = 0.0     # (nh, D, n)
        out.encoder_v[..., idx] = 0.0   # (nh, D, n)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_ablate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add model/src/bdh_surgery/ablate.py model/tests/test_ablate.py
git commit -m "feat(model): collision-neuron identification, ablation, random baseline"
```

---

### Task 9: The sweep

**Files:**
- Create: `model/src/bdh_surgery/sweep.py`
- Test: `model/tests/test_sweep.py`

**Interfaces:**
- Consumes: Tasks 3–8.
- Produces: `THETAS: tuple[float, ...]` (11 values), `SEEDS = (0, 1, 2)`; `run_pair(theta, seed, steps_base, steps_ft) -> dict`; `run_sweep(out_csv: Path, thetas=THETAS, seeds=SEEDS, **kw) -> Path`; CSV columns `theta, seed, m_tau50, m_tau70, m_tau90, m_mean, d_a, d_b, d_mean, d_avg_arm, loss_parent_a, loss_parent_b, self_merge_d, params_merged`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_sweep.py
import csv
from bdh_surgery.sweep import THETAS, SEEDS, run_pair, run_sweep

FAST = dict(steps_base=8, steps_ft=8)


def test_theta_grid_is_eleven_points_and_seeds_are_three():
    assert len(THETAS) == 11 and THETAS[0] == 0.0 and THETAS[-1] == 1.0
    assert len(SEEDS) == 3


def test_run_pair_returns_every_expected_column():
    row = run_pair(0.5, seed=0, **FAST)
    for key in ("theta", "seed", "m_tau50", "m_mean", "d_a", "d_b",
                "d_mean", "d_avg_arm", "self_merge_d", "params_merged"):
        assert key in row, f"missing column {key}"
    assert row["params_merged"] == 405504


def test_run_sweep_writes_one_row_per_pair(tmp_path):
    out = run_sweep(tmp_path / "runs.csv", thetas=(0.0, 1.0), seeds=(0, 1), **FAST)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    assert {float(r["theta"]) for r in rows} == {0.0, 1.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.sweep'`.

- [ ] **Step 3: Write the implementation**

Bases are θ-independent, so they are cached per seed: 3 bases, 66 fine-tunes rather than 33 bases.

```python
# model/src/bdh_surgery/sweep.py
import copy
import csv
from functools import lru_cache
from pathlib import Path

from .ablate import collision_scores  # noqa: F401  (imported for Task 15 reuse)
from .bdh import neuron_axis
from .damage import measure_damage
from .domains import BOS_A, BOS_B, build_lexicons, make_corpus, make_pivot_corpus
from .merge import merge_average, merge_concat
from .overlap import TAUS, matched_correlations, mean_matched_correlation, overlap_at
from .train import finetune, train_base

THETAS = tuple(round(i / 10, 1) for i in range(11))
SEEDS = (0, 1, 2)
PROBE_SEQS = 16

FIELDS = ["theta", "seed", "m_tau50", "m_tau70", "m_tau90", "m_mean",
          "d_a", "d_b", "d_mean", "d_avg_arm", "loss_parent_a", "loss_parent_b",
          "self_merge_d", "params_merged"]


@lru_cache(maxsize=None)
def _base(seed: int, steps: int):
    return train_base(seed=seed, steps=steps)


def run_pair(theta: float, seed: int, steps_base: int = 400,
             steps_ft: int = 600) -> dict:
    base = _base(seed, steps_base)
    lex_a, lex_b = build_lexicons(theta, seed=seed)
    pa = finetune(base, lex_a, BOS_A, seed=seed, steps=steps_ft)
    pb = finetune(base, lex_b, BOS_B, seed=seed, steps=steps_ft)

    probe = make_pivot_corpus(PROBE_SEQS, seed=seed + 30_000)
    matched = matched_correlations(pa, pb, probe)

    concat = merge_concat(pa, pb)
    average = merge_average(pa, pb)
    d_concat = measure_damage(concat, pa, pb, lex_a, lex_b, seed)
    d_average = measure_damage(average, pa, pb, lex_a, lex_b, seed)
    d_self = measure_damage(merge_concat(pa, copy.deepcopy(pa)), pa, pa,
                            lex_a, lex_a, seed)

    return {
        "theta": theta, "seed": seed,
        "m_tau50": overlap_at(matched, TAUS[0]),
        "m_tau70": overlap_at(matched, TAUS[1]),
        "m_tau90": overlap_at(matched, TAUS[2]),
        "m_mean": mean_matched_correlation(matched),
        "d_a": d_concat.d_a, "d_b": d_concat.d_b, "d_mean": d_concat.d_mean,
        "d_avg_arm": d_average.d_mean,
        "loss_parent_a": d_concat.loss_parent_a,
        "loss_parent_b": d_concat.loss_parent_b,
        "self_merge_d": d_self.d_mean,
        "params_merged": sum(p.numel() for p in concat.parameters()),
    }


def run_sweep(out_csv: Path, thetas=THETAS, seeds=SEEDS, **kw) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for theta in thetas:
            for seed in seeds:
                row = run_pair(theta, seed, **kw)
                writer.writerow(row)
                fh.flush()
                print(f"theta={theta:.1f} seed={seed} "
                      f"M={row['m_mean']:.3f} D={row['d_mean']:+.4f}", flush=True)
    return out_csv


if __name__ == "__main__":
    run_sweep(Path(__file__).resolve().parents[3] / "artifacts" / "runs.csv")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_sweep.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the real sweep**

Run: `cd model && uv run python -m bdh_surgery.sweep`
Expected: 33 progress lines, `artifacts/runs.csv` with 33 rows. Time-box to one hour; if slower, cut `THETAS` to 6 points before cutting seeds.

- [ ] **Step 6: Commit**

```bash
git add model/src/bdh_surgery/sweep.py model/tests/test_sweep.py artifacts/runs.csv
git commit -m "feat(model): 33-pair overlap sweep with averaging and self-merge controls"
```

---

### Task 10: Export and the data contract

Freezing this schema is what lets the frontend be built against a fixture while models are still training.

**Files:**
- Create: `model/src/bdh_surgery/export.py`
- Test: `model/tests/test_export.py`

**Schema.** `manifest.json`:

```json
{
  "schema_version": 1,
  "config": {"n_layer": 4, "n_embd": 64, "n_head": 1, "vocab_size": 96, "n": 1024},
  "dtype": "float32",
  "tensors": {
    "decoder":   {"shape": [1024, 64],    "file": "th0.0_s0_A.decoder.bin",   "concat_axis": 0},
    "encoder":   {"shape": [1, 64, 1024], "file": "th0.0_s0_A.encoder.bin",   "concat_axis": 2},
    "encoder_v": {"shape": [1, 64, 1024], "file": "th0.0_s0_A.encoder_v.bin", "concat_axis": 2},
    "attn.freqs":{"shape": [1, 1, 1, 1024],"file": "th0.0_s0_A.freqs.bin",    "concat_axis": 3},
    "lm_head":   {"shape": [64, 96],      "file": "th0.0_s0_A.lm_head.bin",   "concat_axis": null},
    "embed.weight": {"shape": [96, 64],   "file": "th0.0_s0_A.embed.bin",     "concat_axis": null}
  }
}
```

`concat_axis: null` means average. The frontend must never re-derive this classification.

**Interfaces:**
- Consumes: Tasks 1, 5, 9.
- Produces: `FEATURED_THETAS = (0.0, 0.5, 1.0)`; `export_model(net, out_dir, tag) -> dict` (returns the manifest fragment); `export_all(runs_csv, out_dir, featured=FEATURED_THETAS, seed=0) -> Path`; writes `manifest.json`, `sweep.json`, `probes.json`, `*.bin`.

- [ ] **Step 1: Write the failing test**

```python
# model/tests/test_export.py
import json
import numpy as np
import torch
from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.merge import AVERAGE_TENSORS, CONCAT_TENSORS
from bdh_surgery.export import export_model


def test_manifest_classifies_every_tensor_the_merge_rule_knows(tmp_path):
    frag = export_model(BDH(TOY).eval(), tmp_path, tag="t0")
    assert set(frag) == set(CONCAT_TENSORS) | set(AVERAGE_TENSORS)
    for name, axis in CONCAT_TENSORS.items():
        assert frag[name]["concat_axis"] == axis
    for name in AVERAGE_TENSORS:
        assert frag[name]["concat_axis"] is None


def test_binaries_round_trip_exactly(tmp_path):
    net = BDH(TOY).eval()
    frag = export_model(net, tmp_path, tag="t0")
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    for name, meta in frag.items():
        raw = np.fromfile(tmp_path / meta["file"], dtype=np.float32)
        assert list(raw.reshape(meta["shape"]).shape) == meta["shape"]
        assert np.allclose(raw, tensors[name].detach().numpy().ravel(), atol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bdh_surgery.export'`.

- [ ] **Step 3: Write the implementation**

```python
# model/src/bdh_surgery/export.py
"""Writes the static data the browser loads. Float32 for exactness; quantize
only if load time is measured to be a problem.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .bdh import BDH, TOY, neuron_axis
from .domains import BOS_A, BOS_B, build_lexicons, make_corpus, make_pivot_corpus
from .merge import AVERAGE_TENSORS, CONCAT_TENSORS
from .train import finetune, train_base

FEATURED_THETAS = (0.0, 0.5, 1.0)
SCHEMA_VERSION = 1
_FILE_STEM = {"attn.freqs": "freqs", "embed.weight": "embed"}


def export_model(net: BDH, out_dir: Path, tag: str) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    frag = {}
    for name in list(CONCAT_TENSORS) + list(AVERAGE_TENSORS):
        arr = tensors[name].detach().cpu().numpy().astype(np.float32)
        fname = f"{tag}.{_FILE_STEM.get(name, name)}.bin"
        arr.tofile(out_dir / fname)
        frag[name] = {"shape": list(arr.shape), "file": fname,
                      "concat_axis": CONCAT_TENSORS.get(name)}
    return frag


def export_all(runs_csv: Path, out_dir: Path,
               featured=FEATURED_THETAS, seed: int = 0,
               steps_base: int = 400, steps_ft: int = 600) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = train_base(seed=seed, steps=steps_base)
    manifest = {"schema_version": SCHEMA_VERSION, "dtype": "float32",
                "config": {"n_layer": TOY.n_layer, "n_embd": TOY.n_embd,
                           "n_head": TOY.n_head, "vocab_size": TOY.vocab_size,
                           "n": neuron_axis(TOY)},
                "featured": {}}
    for theta in featured:
        lex_a, lex_b = build_lexicons(theta, seed=seed)
        tag = f"th{theta}_s{seed}"
        manifest["featured"][str(theta)] = {
            "lex_a": lex_a, "lex_b": lex_b,
            "A": export_model(finetune(base, lex_a, BOS_A, seed, steps_ft),
                              out_dir, f"{tag}_A"),
            "B": export_model(finetune(base, lex_b, BOS_B, seed, steps_ft),
                              out_dir, f"{tag}_B"),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = list(csv.DictReader(Path(runs_csv).open()))
    (out_dir / "sweep.json").write_text(json.dumps(
        {"points": [{k: (float(v) if k != "seed" else int(v))
                     for k, v in r.items()} for r in rows]}, indent=2))

    probe = make_pivot_corpus(8, seed=seed + 30_000)
    lex_a, lex_b = build_lexicons(featured[0], seed=seed)
    (out_dir / "probes.json").write_text(json.dumps({
        "pivot": probe.tolist(),
        "eval_a": make_corpus(lex_a, BOS_A, 8, seed=seed + 10_000).tolist(),
        "eval_b": make_corpus(lex_b, BOS_B, 8, seed=seed + 20_000).tolist(),
    }, indent=2))
    return out_dir / "manifest.json"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    export_all(root / "artifacts" / "runs.csv", root / "web" / "public" / "data")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_export.py -v`
Expected: 2 passed.

- [ ] **Step 5: Export and commit**

```bash
cd model && uv run python -m bdh_surgery.export && cd ..
git add model/src/bdh_surgery/export.py model/tests/test_export.py web/public/data/
git commit -m "feat(model): export weights, sweep and probes on frozen v1 schema"
```

---

### Task 11: JavaScript forward pass

**Files:**
- Create: `web/src/bdh_forward.js` (replace stub), `web/src/merge.js`, `model/tests/test_js_parity.py`

**Interfaces:**
- Consumes: `manifest.json` schema from Task 10.
- Produces: `loadModel(manifest, side, fetchBin) -> Promise<Model>`; `mergeConcat(a, b) -> Model`; `mergeAverage(a, b) -> Model`; `forward(model, tokenIds, {ablated}) -> Float32Array[]` returning per-position logits; `Model` is `{cfg, n, tensors: {decoder, encoder, encoder_v, freqs, lm_head, embed}}` with every tensor a `Float32Array`.

- [ ] **Step 1: Write the failing parity test**

```python
# model/tests/test_js_parity.py
"""Runs the browser forward pass under node and compares against PyTorch."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from bdh_surgery.bdh import BDH, TOY
from bdh_surgery.export import export_model
from bdh_surgery.recurrent import recurrent_logits

WEB = Path(__file__).resolve().parents[2] / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_forward_matches_pytorch():
    torch.manual_seed(0)
    net = BDH(TOY).eval()
    tokens = [3, 17, 42, 8, 1, 75, 30]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frag = export_model(net, tmp, tag="parity")
        (tmp / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "dtype": "float32",
            "config": {"n_layer": TOY.n_layer, "n_embd": TOY.n_embd,
                       "n_head": TOY.n_head, "vocab_size": TOY.vocab_size, "n": 1024},
            "featured": {"0.0": {"A": frag, "B": frag, "lex_a": [], "lex_b": []}},
        }))
        script = f"""
        import {{ loadModelFromDir, forward }} from '{(WEB / "src/bdh_forward.js").as_posix()}';
        const m = await loadModelFromDir('{tmp.as_posix()}', '0.0', 'A');
        const out = forward(m, {tokens});
        console.log(JSON.stringify(Array.from(out[out.length - 1])));
        """
        js = tmp / "parity.mjs"; js.write_text(script)
        res = subprocess.run(["node", str(js)], capture_output=True, text=True,
                             check=True)
        got = torch.tensor(json.loads(res.stdout.strip()))
    want = recurrent_logits(net, torch.tensor([tokens]))[0, -1]
    assert (got - want).abs().max().item() < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/test_js_parity.py -v`
Expected: FAIL — `loadModelFromDir` is not exported (or node reports a missing module).

- [ ] **Step 3: Write the implementation**

Port `recurrent.py` line for line. `ln` is parameter-free LayerNorm over the D axis.

```javascript
// web/src/bdh_forward.js
// Recurrent BDH-GPU forward pass (Eq. 8, arXiv:2509.26507), ported from
// model/src/bdh_surgery/recurrent.py and checked against it by test_js_parity.py.

const TENSOR_FILES = {
  decoder: 'decoder', encoder: 'encoder', encoder_v: 'encoder_v',
  'attn.freqs': 'freqs', lm_head: 'lm_head', 'embed.weight': 'embed',
};

export async function loadModel(manifest, theta, side, fetchBin) {
  const frag = manifest.featured[theta][side];
  const tensors = {};
  for (const name of Object.keys(TENSOR_FILES)) {
    const meta = frag[name];
    tensors[name] = { data: await fetchBin(meta.file), shape: meta.shape,
                      concatAxis: meta.concat_axis };
  }
  return { cfg: manifest.config, n: manifest.config.n, tensors };
}

export async function loadModelFromDir(dir, theta, side) {
  const fs = await import('node:fs/promises');
  const manifest = JSON.parse(await fs.readFile(`${dir}/manifest.json`, 'utf8'));
  return loadModel(manifest, theta, side, async (file) => {
    const buf = await fs.readFile(`${dir}/${file}`);
    return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  });
}

function layerNorm(v) {
  const d = v.length;
  let mean = 0; for (let i = 0; i < d; i++) mean += v[i]; mean /= d;
  let varr = 0; for (let i = 0; i < d; i++) { const t = v[i] - mean; varr += t * t; }
  const inv = 1 / Math.sqrt(varr / d + 1e-5);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = (v[i] - mean) * inv;
  return out;
}

// x (D) @ encoder (1, D, n) -> (n), then ReLU
function encodeRelu(x, enc, D, n) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let acc = 0;
    for (let i = 0; i < D; i++) acc += x[i] * enc[i * n + j];
    out[j] = acc > 0 ? acc : 0;
  }
  return out;
}

// RoPE over adjacent pairs on the n axis, matching Attention.rope
function rope(v, freqs, t, n) {
  const out = new Float32Array(n);
  for (let j = 0; j < n; j += 2) {
    const ph = ((t * freqs[j]) % 1) * 2 * Math.PI;
    const c = Math.cos(ph), s = Math.sin(ph);
    out[j] = v[j] * c - v[j + 1] * s;
    out[j + 1] = v[j + 1] * c + v[j] * s;
  }
  return out;
}

export function forward(model, tokenIds, { ablated = null } = {}) {
  const { n_layer: L, n_embd: D, vocab_size: V } = model.cfg;
  const n = model.n;
  const T = model.tensors;
  const mask = new Float32Array(n).fill(1);
  if (ablated) for (const i of ablated) mask[i] = 0;

  const rho = [];
  for (let l = 0; l < L; l++) rho.push(new Float32Array(n * D));
  const outputs = [];

  for (let t = 0; t < tokenIds.length; t++) {
    let x = layerNorm(T['embed.weight'].data.slice(tokenIds[t] * D, tokenIds[t] * D + D));
    for (let l = 0; l < L; l++) {
      const xs = encodeRelu(x, T.encoder.data, D, n);
      for (let j = 0; j < n; j++) xs[j] *= mask[j];
      const qr = rope(xs, T['attn.freqs'].data, t, n);

      const yKV = new Float32Array(D);                       // read before write
      for (let j = 0; j < n; j++) {
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yKV[i] += q * rho[l][off + i];
      }
      for (let j = 0; j < n; j++) {                          // then write
        const q = qr[j]; if (q === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) rho[l][off + i] += q * x[i];
      }

      const ys = encodeRelu(layerNorm(yKV), T.encoder_v.data, D, n);
      const yMLP = new Float32Array(D);
      for (let j = 0; j < n; j++) {
        const xy = xs[j] * ys[j]; if (xy === 0) continue;
        const off = j * D;
        for (let i = 0; i < D; i++) yMLP[i] += xy * T.decoder.data[off + i];
      }
      const y = layerNorm(yMLP);
      const sum = new Float32Array(D);
      for (let i = 0; i < D; i++) sum[i] = x[i] + y[i];
      x = layerNorm(sum);
    }
    const logits = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let acc = 0;
      for (let i = 0; i < D; i++) acc += x[i] * T.lm_head.data[i * V + v];
      logits[v] = acc;
    }
    outputs.push(logits);
  }
  return outputs;
}
```

```javascript
// web/src/merge.js
// Client-side merge. Classification comes from the manifest's concat_axis,
// never re-derived here — that is what keeps it in step with merge.py.
function catAxis(a, b, shape, axis) {
  if (axis === shape.length - 1) {           // interleave along the last axis
    const inner = shape[axis], rows = a.length / inner;
    const out = new Float32Array(a.length + b.length);
    for (let r = 0; r < rows; r++) {
      out.set(a.subarray(r * inner, (r + 1) * inner), r * 2 * inner);
      out.set(b.subarray(r * inner, (r + 1) * inner), r * 2 * inner + inner);
    }
    return out;
  }
  const out = new Float32Array(a.length + b.length);  // axis 0: plain append
  out.set(a, 0); out.set(b, a.length);
  return out;
}

function average(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] + b[i]) / 2;
  return out;
}

function blend(a, b, concat) {
  const tensors = {};
  for (const name of Object.keys(a.tensors)) {
    const ta = a.tensors[name], tb = b.tensors[name];
    const axis = ta.concatAxis;
    if (concat && axis !== null) {
      const shape = ta.shape.slice();
      shape[axis] *= 2;
      tensors[name] = { data: catAxis(ta.data, tb.data, ta.shape, axis),
                        shape, concatAxis: axis };
    } else {
      tensors[name] = { data: average(ta.data, tb.data),
                        shape: ta.shape.slice(), concatAxis: axis };
    }
  }
  const n = concat ? a.n * 2 : a.n;
  return { cfg: { ...a.cfg, n }, n, tensors };
}

export const mergeConcat = (a, b) => blend(a, b, true);
export const mergeAverage = (a, b) => blend(a, b, false);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/test_js_parity.py -v`
Expected: PASS. If node is absent the test skips — install node rather than accepting the skip.

- [ ] **Step 5: Commit**

```bash
git add web/src/bdh_forward.js web/src/merge.js model/tests/test_js_parity.py
git commit -m "feat(web): recurrent forward pass and client-side merge, parity-tested"
```

---

### Task 12: Web scaffold, Acts 1 and 2

**Files:**
- Create: `web/package.json`, `web/vite.config.js`, `web/src/main.js`, `web/src/App.svelte`, `web/src/lib/NeuronStrip.svelte`, `web/src/lib/OutputPanels.svelte`, `web/src/lib/store.js`
- Modify: `web/index.html`

**Interfaces:**
- Consumes: Task 11 `loadModel`, `mergeConcat`, `mergeAverage`, `forward`.
- Produces: `store.js` exporting Svelte stores `manifest`, `theta`, `mergeMode` (`'concat' | 'average'`), `ablated` (array of indices), and derived `mergedModel`; `NeuronStrip` props `{n, splitAt, highlight}`; `OutputPanels` props `{parentA, parentB, merged, oracle}`.

- [ ] **Step 1: Scaffold and verify the build**

```bash
cd web && npm create vite@latest . -- --template svelte && npm install
npm run build
```

Expected: `dist/` produced with no errors. Add `node_modules/` and `dist/` to `.gitignore` if absent.

- [ ] **Step 2: Write the fixture so the UI is unblocked**

Create `web/public/data/manifest.json` by hand (or from Task 10) before any component is written. The frontend is built against this file, not against trained models.

- [ ] **Step 3: Build Acts 1 and 2**

`App.svelte` loads the manifest on mount, loads parents A and B for the selected θ, computes the merged model reactively from `mergeMode`, and renders `NeuronStrip` plus `OutputPanels`. The concat/average control is a two-button segmented toggle bound to `mergeMode`. `NeuronStrip` renders to a `<canvas>`: neurons `0..splitAt` in the parent-A colour, `splitAt..n` in parent-B's; under `average` the strip is drawn at width `n` in a blended colour. `OutputPanels` shows four columns — parent A, parent B, merged, oracle — generated from the same prompt via `forward`.

- [ ] **Step 4: Verify in the browser**

Run: `cd web && npm run dev`
Expected: page opens with a preset already running, no Run button. Toggling concat→average visibly degrades merged output while parents stay unchanged. Toggle responds in under a second.

- [ ] **Step 5: Commit**

```bash
git add web/ && git commit -m "feat(web): Vite+Svelte scaffold, Act 1 fuse and Act 2 concat/average toggle"
```

---

### Task 13: Act 3 — the phase diagram

**Files:**
- Create: `web/src/lib/PhaseDiagram.svelte`
- Modify: `web/src/App.svelte`

**Interfaces:**
- Consumes: `sweep.json` from Task 10; `theta` store from Task 12.
- Produces: `PhaseDiagram` props `{points, xKey, yKey, marker}` and event `select` carrying the chosen θ.

- [ ] **Step 1: Render the curve**

Hand-rolled inline SVG. x = `m_mean` (measured overlap), y = `d_mean`. Group `sweep.json` points by θ, plot the mean across the three seeds, and draw a vertical error bar from min to max seed value at each point. Plot `d_avg_arm` as a second, visually muted series — the averaging control.

- [ ] **Step 2: Wire the slider**

A θ slider snaps to the 11 grid values, moves the marker, and dispatches `select`. `App.svelte` loads that θ's featured weights when one is available and otherwise shows the point's precomputed metrics with a "precomputed" badge.

- [ ] **Step 3: Verify**

Run: `cd web && npm run dev`
Expected: dragging the slider moves the marker; every point shows an error bar; the averaging series sits visibly above the concat series.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/PhaseDiagram.svelte web/src/App.svelte
git commit -m "feat(web): Act 3 live phase diagram with seed error bars"
```

---

### Task 14: Act 4 — surgery

**Files:**
- Create: `web/src/lib/Surgery.svelte`, `web/src/collision.js`
- Modify: `web/src/App.svelte`

**Interfaces:**
- Consumes: Task 11 `forward`; Task 12 `ablated` store.
- Produces: `collisionScores(model, probeA, probeB) -> Float32Array` and `topK(scores, k) -> number[]`, both porting `ablate.py`'s formula exactly; `Surgery` props `{model, probes}`.

- [ ] **Step 1: Port the collision score**

`collision.js` mirrors `ablate.py`: `joint = min(meanActA, meanActB)`, `imbalance = |a−b| / (a+b+1e-8)`, `score = joint * (1 − imbalance)`. Activations come from instrumenting `forward` to also return per-neuron `xy` means.

- [ ] **Step 2: Build the UI**

Canvas heat-map of the merged neuron axis, collision neurons highlighted. Controls: a `k` slider, an "ablate collision set" button, and an "ablate k random instead" button. Both write into the `ablated` store, which `forward` already honours. Output panels re-render live.

- [ ] **Step 3: Verify the control is visible**

Expected: ablating the collision set and ablating k random neurons are shown side by side, never one alone. The recovery number displayed is the **difference** between them.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/Surgery.svelte web/src/collision.js web/src/App.svelte
git commit -m "feat(web): Act 4 collision-neuron ablation with random baseline"
```

---

### Task 15: Narrative, honesty labels, and documentation

**Files:**
- Create: `web/src/lib/Narrative.svelte`, `web/src/lib/HonestyBadge.svelte`
- Modify: `README.md`, `PROVENANCE.md`, `web/src/App.svelte`

- [ ] **Step 1: Add the guided narrative**

`Narrative.svelte` steps the learner through Acts 1→4 with a short paragraph each, then unlocks every control for free exploration. The page opens mid-Act-1 with a preset already running.

- [ ] **Step 2: Label every element**

`HonestyBadge` renders one of `live` / `precomputed` / `illustration`. Apply: `live` on output panels, neuron strip and ablation; `precomputed` on the phase diagram and parent weights; `illustration` on any explanatory diagram. Add a persistent footer stating the caps — `n = 1024` per parent, `2048` merged, sequence length, and "toy-scale reimplementation, not an official BDH model".

- [ ] **Step 3: Write the README**

Cover, in order: the one-sentence claim; intended learner and prerequisites; learning objectives; artifact architecture; the role of every major component; the live/precomputed/synthetic/illustration table; how to reproduce (`uv run pytest`, `uv run python -m bdh_surgery.sweep`, `uv run python -m bdh_surgery.export`, `npm run build`); credits and licences.

- [ ] **Step 4: Complete PROVENANCE.md**

Record: `bdh.py` adapted from pathwaycom/bdh (MIT, verified via GitHub API); synthetic data generated by `domains.py`, not scraped; weights trained in-repo with seeds pinned in `sweep.py`; the AI-assistance disclosure.

- [ ] **Step 5: Full verification**

```bash
cd model && uv run pytest -v && cd ../web && npm run build
```

Expected: all tests pass; `dist/` builds clean.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: guided narrative, honesty labels, README and provenance"
```

---

## Self-Review

**Spec coverage.** §2.1 domains → Task 3. §2.2 θ vs M → Tasks 3, 6, 9 (both recorded per row). §2.3 Hungarian overlap → Task 6. §2.4 damage → Task 7. §2.5 sweep → Task 9. §2.6 controls → averaging arm and self-merge in Task 9, random-ablation in Task 8, **parameter-count control is deferred** (spec §8 decision 6 allows dropping it if time is short; add a 34th sweep row training one parent at `multiplier=32` if time allows). §2.7 localization → Tasks 8, 14. §3.1 four acts → Tasks 12–14. §3.2 live-vs-precomputed → Task 15. §3.3 truth beside estimate → Task 12 `OutputPanels`. §4 stack → Global Constraints. §4.3 data contract → Task 10.

**Open decisions now closed.** θ = surface-form overlap (§2.1). Data contract = manifest v1 + float32 `.bin` (§4.3). Framework = Svelte (§4.2). Quantization = float32, revisit only on measured load time. τ = report at 0.5/0.7/0.9 plus thresholdless mean.

**Type consistency.** `neuron_axis` used identically in Tasks 1, 5, 6, 8, 10. `CONCAT_TENSORS`/`AVERAGE_TENSORS` defined in Task 5, consumed in Tasks 10, 11. `Damage` fields match between Tasks 7 and 9. Manifest `concat_axis` written in Task 10 and read in Task 11 without re-derivation. `forward(model, tokenIds, {ablated})` signature consistent across Tasks 11, 12, 14.

**Known risk carried forward.** Task 9's sweep is the long pole. If it exceeds one hour, cut θ from 11 points to 6 — never cut seeds, since without seed variance the phase diagram is not evidence.
