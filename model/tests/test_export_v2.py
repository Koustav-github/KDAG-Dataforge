import csv
import json

from bdh_surgery.datasets import BASELINE, LARGE_VOCAB, LONG_PHRASE
from bdh_surgery.export import export_all_datasets, export_dataset, export_probes

FAST = dict(featured=(0.0, 1.0), steps_base=8, steps_ft=8)


def test_export_dataset_returns_the_v2_fragment_shape(tmp_path):
    frag = export_dataset(LONG_PHRASE, tmp_path, **FAST)
    assert frag["label"] == LONG_PHRASE.label
    assert frag["n_concepts"] == LONG_PHRASE.n_concepts
    assert frag["phrase_len"] == LONG_PHRASE.phrase_len
    assert frag["config"]["n"] == 1024
    assert set(frag["featured"]) == {"0.0", "1.0"}
    for theta in frag["featured"].values():
        assert set(theta) == {"lex_a", "lex_b", "A", "B"}


def test_large_vocab_export_uses_its_own_widened_vocab_size(tmp_path):
    frag = export_dataset(LARGE_VOCAB, tmp_path, **FAST)
    assert frag["config"]["vocab_size"] == 128


def test_export_probes_are_scoped_to_the_dataset(tmp_path):
    p = export_probes(LARGE_VOCAB, tmp_path, featured=(0.0, 1.0))
    assert set(p["featured"]) == {"0.0", "1.0"}
    # tokens must stay inside THIS dataset's vocab, not baseline's narrower one
    flat = [t for row in p["featured"]["1.0"]["eval_a"] for t in row]
    assert max(flat) < LARGE_VOCAB.vocab_size


def test_export_all_datasets_nests_every_dataset_under_one_manifest(tmp_path):
    for spec, name in ((BASELINE, "baseline"), (LONG_PHRASE, "long_phrase")):
        rows = [{"dataset": spec.id, "theta": "0.5", "seed": "0", "m_mean": "0.7",
                 "d_mean": "0.5"}]
        csv_path = tmp_path / f"runs_{name}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    out = tmp_path / "data"
    manifest_path = export_all_datasets(
        {"baseline": tmp_path / "runs_baseline.csv",
         "long_phrase": tmp_path / "runs_long_phrase.csv"},
        out, specs=(BASELINE, LONG_PHRASE), **FAST)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert set(manifest["datasets"]) == {"baseline", "long_phrase"}

    sweep = json.loads((out / "sweep.json").read_text())
    assert sweep["datasets"]["baseline"]["points"][0]["theta"] == 0.5
    assert "dataset" not in sweep["datasets"]["baseline"]["points"][0]

    probes = json.loads((out / "probes.json").read_text())
    assert set(probes["datasets"]) == {"baseline", "long_phrase"}


def test_baseline_runs_csv_without_a_dataset_column_still_exports(tmp_path):
    """The real baseline runs.csv predates the `dataset` column entirely."""
    rows = [{"theta": "0.5", "seed": "0", "m_mean": "0.7", "d_mean": "0.5"}]
    csv_path = tmp_path / "runs.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    manifest_path = export_all_datasets(
        {"baseline": csv_path}, tmp_path / "data", specs=(BASELINE,), **FAST)
    sweep = json.loads((manifest_path.parent / "sweep.json").read_text())
    assert sweep["datasets"]["baseline"]["points"][0]["theta"] == 0.5
