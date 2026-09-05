"""Writes the static data the browser loads. Float32 for exactness; quantize
only if load time is measured to be a problem.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .bdh import BDH, TOY, neuron_axis
from .domains import BOS_A, BOS_B, build_lexicons, make_corpus, make_pivot_corpus
from .merge import AVERAGE_TENSORS, CONCAT_TENSORS
from .train import finetune, train_base

# Every theta the sweep covers. At float32 this would be ~18 MB; at int8 it is
# ~4.6 MB, which is what makes the artifact live at all 11 instead of 3.
FEATURED_THETAS = tuple(round(i / 10, 1) for i in range(11))
EXPORT_DTYPE = "int8"
SCHEMA_VERSION = 1
_FILE_STEM = {"attn.freqs": "freqs", "embed.weight": "embed"}


def export_model(net: BDH, out_dir: Path, tag: str, dtype: str = "float32") -> dict:
    """Write one model's tensors as raw binary plus a manifest fragment.

    float32 is exact and is what the JS/Python parity test uses. int8 is a
    symmetric per-tensor quantization (scale = max|w| / 127) that cuts transfer
    4x; the browser dequantizes once at load time into Float32Array, so the
    forward pass never sees a quantized value and needs no changes. Shipping
    int8 is what makes all 11 swept theta affordable to include instead of 3.
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


def export_all(runs_csv: Path, out_dir: Path,
               featured=FEATURED_THETAS, seed: int = 0,
               steps_base: int = 400, steps_ft: int = 600) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = train_base(seed=seed, steps=steps_base)
    manifest = {"schema_version": SCHEMA_VERSION, "dtype": EXPORT_DTYPE,
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
                     for k, v in r.items()} for r in rows]}, indent=2))

    # Eval corpora are per-theta. Each theta has its OWN target lexicons, so a
    # single shared probe set would score the theta=0.5 and theta=1.0 parents
    # against theta=0.0's vocabulary — measuring the wrong thing entirely. The
    # pivot probe is theta-independent and stays shared.
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
    export_all(root / "artifacts" / "runs.csv", root / "web" / "public" / "data")
