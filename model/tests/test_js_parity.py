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
        import {{ loadModelFromDir, forward }} from '{(WEB / "src/bdh_forward.js").as_uri()}';
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
