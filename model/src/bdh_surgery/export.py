"""Writes the static data the browser loads. Float32 for exactness; quantize
only if load time is measured to be a problem.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .bdh import BDH, TOY, neuron_axis
from .datasets import BASELINE, DATASETS
from .domains import BOS_A, BOS_B, build_lexicons, make_corpus, make_pivot_corpus
from .merge import AVERAGE_TENSORS, CONCAT_TENSORS
from .train import finetune, train_base

# Every theta the sweep covers. At float32 this would be ~18 MB; at int8 it is
# ~4.6 MB, which is what makes the artifact live at all 11 instead of 3.
FEATURED_THETAS = tuple(round(i / 10, 1) for i in range(11))
EXPORT_DTYPE = "int8"
SCHEMA_VERSION = 2
_FILE_STEM = {"attn.freqs": "freqs", "embed.weight": "embed"}


def export_model(net: BDH, out_dir: Path, tag: str, dtype: str = "float32") -> dict:
    """Write one model's tensors as raw binary plus a manifest fragment.

    float32 is exact and is what the JS/Python parity test uses. int8 is a
    symmetric per-tensor quantization (scale = max|w| / 127) that cuts transfer
    4x; the browser dequantizes once at load time into Float32Array, so the
    forward pass never sees a quantized value and needs no changes.
    """
    assert dtype in ("float32", "int8"), dtype
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    frag = {}
    for name in list(CONCAT_TENSORS) + list(AVERAGE_TENSORS):
        arr = tensors[name].detach().cpu().numpy().astype(np.float32)
        fname = f"{tag}.{_FILE_STEM.get(name, name)}.bin"
        meta = {"shape": list(arr.shape), "file": fname,
                "concat_axis": CONCAT_TENSORS.get(name)}
        if dtype == "int8":
            peak = float(np.abs(arr).max())
            scale = peak / 127.0 if peak > 0 else 1.0
            np.clip(np.rint(arr / scale), -127, 127).astype(np.int8).tofile(out_dir / fname)
            meta["dtype"] = "int8"
            meta["scale"] = scale
        else:
            arr.tofile(out_dir / fname)
            meta["dtype"] = "float32"
        frag[name] = meta
    return frag


def export_dataset(spec, out_dir: Path, featured=FEATURED_THETAS, seed: int = 0,
                   steps_base: int = 400, steps_ft: int = 600) -> dict:
    """Train and export the featured-theta models for one dataset.

    Returns the manifest fragment for `datasets[spec.id]`: config, label, and
    the per-theta parent weights. Every dataset goes through the SAME call
    shape (spec is never None here) so there is exactly one export code path —
    train_base/finetune/build_lexicons/make_corpus already treat spec=BASELINE
    as byte-identical to the original spec=None calls (see test_datasets.py),
    so this does not change what baseline's weights are, only how they're made.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lay = spec.layout
    base = train_base(seed=seed, steps=steps_base, spec=None if spec.id == "baseline" else spec)
    fine_spec = None if spec.id == "baseline" else spec

    fragment = {
        "label": spec.label,
        "blurb": spec.blurb,
        "n_concepts": spec.n_concepts,
        "phrase_len": spec.phrase_len,
        "config": {"n_layer": spec.config().n_layer, "n_embd": spec.config().n_embd,
                   "n_head": spec.config().n_head, "vocab_size": spec.config().vocab_size,
                   "n": neuron_axis(spec.config())},
        "featured": {},
    }
    for theta in featured:
        lex_a, lex_b = build_lexicons(theta, seed=seed, n_concepts=spec.n_concepts,
                                      target_base=lay["target_base"])
        tag = f"{spec.id}_th{theta}_s{seed}"
        pa = finetune(base, lex_a, lay["bos_a"], seed, steps_ft, spec=fine_spec)
        pb = finetune(base, lex_b, lay["bos_b"], seed, steps_ft, spec=fine_spec)
        fragment["featured"][str(theta)] = {
            "lex_a": lex_a, "lex_b": lex_b,
            "A": export_model(pa, out_dir, f"{tag}_A", EXPORT_DTYPE),
            "B": export_model(pb, out_dir, f"{tag}_B", EXPORT_DTYPE),
        }
    return fragment


def export_probes(spec, out_dir: Path, featured=FEATURED_THETAS, seed: int = 0) -> dict:
    """Per-theta eval corpora for one dataset. Each theta has its own target
    lexicons, so a single shared probe set would score every theta's parents
    against theta=0.0's vocabulary. The pivot probe is theta-independent within
    a dataset (it never touches target tokens) but DOES depend on the dataset's
    own n_concepts/pivot_base, so it is not shared across datasets either.
    """
    lay = spec.layout
    kw = dict(phrase_len=spec.phrase_len, n_concepts=spec.n_concepts,
              pivot_base=lay["pivot_base"])
    pivot = make_pivot_corpus(8, seed=seed + 30_000, bos=lay["bos_a"], eos=lay["eos"], **kw)
    by_theta = {}
    for theta in featured:
        lex_a, lex_b = build_lexicons(theta, seed=seed, n_concepts=spec.n_concepts,
                                      target_base=lay["target_base"])
        eval_kw = dict(**kw, sep=lay["sep"], eos=lay["eos"])
        by_theta[str(theta)] = {
            "eval_a": make_corpus(lex_a, lay["bos_a"], 8, seed=seed + 10_000, **eval_kw).tolist(),
            "eval_b": make_corpus(lex_b, lay["bos_b"], 8, seed=seed + 20_000, **eval_kw).tolist(),
        }
    return {"pivot": pivot.tolist(), "featured": by_theta}


def export_all_datasets(runs_csvs: dict, out_dir: Path, specs=DATASETS,
                        featured=FEATURED_THETAS, seed: int = 0,
                        steps_base: int = 400, steps_ft: int = 600) -> Path:
    """Schema v2: every dataset nested under datasets[spec.id], instead of the
    v1 layout that could only ever describe one dataset at a time.

    `runs_csvs` maps spec.id -> path to that dataset's sweep CSV.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "dtype": EXPORT_DTYPE, "datasets": {}}
    sweep = {"datasets": {}}
    probes = {"datasets": {}}

    for spec in specs:
        print(f"[export] {spec.id}: training {len(featured)} featured thetas...", flush=True)
        manifest["datasets"][spec.id] = export_dataset(
            spec, out_dir, featured=featured, seed=seed,
            steps_base=steps_base, steps_ft=steps_ft)
        probes["datasets"][spec.id] = export_probes(spec, out_dir, featured=featured, seed=seed)

        csv_path = Path(runs_csvs[spec.id])
        rows = list(csv.DictReader(csv_path.open()))
        points = []
        for r in rows:
            point = {k: (float(v) if k not in ("seed", "dataset") else v)
                    for k, v in r.items() if k != "dataset"}
            point["seed"] = int(point["seed"])
            points.append(point)
        sweep["datasets"][spec.id] = {"points": points}
        print(f"[export] {spec.id}: done ({len(points)} sweep rows)", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "sweep.json").write_text(json.dumps(sweep, indent=2))
    (out_dir / "probes.json").write_text(json.dumps(probes, indent=2))
    return out_dir / "manifest.json"


def export_all(runs_csv: Path, out_dir: Path,
               featured=FEATURED_THETAS, seed: int = 0,
               steps_base: int = 400, steps_ft: int = 600) -> Path:
    """Deprecated schema-v1 path (single dataset, no `datasets` nesting).
    Kept only because model/tests/test_export.py exercises it directly against
    a fresh tmp model; the live artifact is built by export_all_datasets now.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = train_base(seed=seed, steps=steps_base)
    manifest = {"schema_version": 1, "dtype": EXPORT_DTYPE,
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
                              out_dir, f"{tag}_A", EXPORT_DTYPE),
            "B": export_model(finetune(base, lex_b, BOS_B, seed, steps_ft),
                              out_dir, f"{tag}_B", EXPORT_DTYPE),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows = list(csv.DictReader(Path(runs_csv).open()))
    (out_dir / "sweep.json").write_text(json.dumps(
        {"points": [{k: (float(v) if k != "seed" else int(v))
                     for k, v in r.items() if k != "dataset"}
                    for r in rows]}, indent=2))

    probe = make_pivot_corpus(8, seed=seed + 30_000)
    by_theta = {}
    for theta in featured:
        lex_a, lex_b = build_lexicons(theta, seed=seed)
        by_theta[str(theta)] = {
            "eval_a": make_corpus(lex_a, BOS_A, 8, seed=seed + 10_000).tolist(),
            "eval_b": make_corpus(lex_b, BOS_B, 8, seed=seed + 20_000).tolist(),
        }
    (out_dir / "probes.json").write_text(json.dumps({
        "pivot": probe.tolist(),
        "featured": by_theta,
    }, indent=2))
    return out_dir / "manifest.json"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    export_all_datasets(
        {"baseline": root / "artifacts" / "runs.csv",
         "long_phrase": root / "artifacts" / "runs_long_phrase.csv",
         "large_vocab": root / "artifacts" / "runs_large_vocab.csv"},
        root / "web" / "public" / "data",
    )
