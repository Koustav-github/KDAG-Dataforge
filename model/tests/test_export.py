import json
import numpy as np
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


def test_int8_export_round_trips_within_quantization_error(tmp_path):
    """The browser dequantizes as value * scale, so check that reconstruction
    lands within one quantization step of the original weights."""
    net = BDH(TOY).eval()
    frag = export_model(net, tmp_path, tag="q", dtype="int8")
    tensors = dict(net.named_parameters()) | {"attn.freqs": net.attn.freqs}
    for name, meta in frag.items():
        assert meta["dtype"] == "int8"
        raw = np.fromfile(tmp_path / meta["file"], dtype=np.int8)
        original = tensors[name].detach().numpy().ravel()
        assert raw.size == original.size
        recon = raw.astype(np.float32) * meta["scale"]
        # symmetric per-tensor quantization: worst case is half a step
        assert np.abs(recon - original).max() <= meta["scale"] * 0.5 + 1e-6


def test_int8_export_is_a_quarter_the_size_of_float32(tmp_path):
    net = BDH(TOY).eval()
    f32 = export_model(net, tmp_path / "f", tag="f", dtype="float32")
    i8 = export_model(net, tmp_path / "i", tag="i", dtype="int8")
    big = sum((tmp_path / "f" / m["file"]).stat().st_size for m in f32.values())
    small = sum((tmp_path / "i" / m["file"]).stat().st_size for m in i8.values())
    assert small * 4 == big
