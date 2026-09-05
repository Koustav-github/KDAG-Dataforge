"""Runs web/src/merge.js under node and checks its output layout against hand-computed
expectations, so a future refactor that inverts the axis-0/last-axis branches (or drops
the concat_axis-null averaging rule) fails loudly instead of silently.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# Tiny synthetic BDH-shaped model: n_layer=1, n_embd(D)=2, n_head=1, vocab_size(V)=2, n=4.
# Values are chosen to be easily distinguishable by eye/assert: parent A uses small
# integers, parent B is the same pattern offset by +100.
_NODE_SCRIPT = """
import {{ mergeConcat, mergeAverage }} from '{merge_uri}';
import {{ forward }} from '{forward_uri}';

function buildModel(offset) {{
  const decoder = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8].map(v => v + offset));
  // encoder (1, D=2, n=4): row i=0 -> [10,11,12,13], row i=1 -> [20,21,22,23]
  const encoder = new Float32Array([10, 11, 12, 13, 20, 21, 22, 23].map(v => v + offset));
  const encoder_v = new Float32Array([30, 31, 32, 33, 40, 41, 42, 43].map(v => v + offset));
  const freqs = new Float32Array([0.1, 0.1, 0.05, 0.05]);
  const lm_head = new Float32Array([2, 4, 6, 8].map(v => v + offset));
  const embed = new Float32Array([0.1, 0.2, 0.3, 0.4].map(v => v + offset));
  return {{
    cfg: {{ n_layer: 1, n_embd: 2, n_head: 1, vocab_size: 2 }},
    n: 4,
    tensors: {{
      decoder: {{ data: decoder, shape: [4, 2], concatAxis: 0 }},
      encoder: {{ data: encoder, shape: [1, 2, 4], concatAxis: 2 }},
      encoder_v: {{ data: encoder_v, shape: [1, 2, 4], concatAxis: 2 }},
      'attn.freqs': {{ data: freqs, shape: [1, 1, 1, 4], concatAxis: 3 }},
      lm_head: {{ data: lm_head, shape: [2, 2], concatAxis: null }},
      'embed.weight': {{ data: embed, shape: [2, 2], concatAxis: null }},
    }},
  }};
}}

const a = buildModel(0);
const b = buildModel(100);

const concatModel = mergeConcat(a, b);
const avgModel = mergeAverage(a, b);

const smokeOut = forward(concatModel, [0, 1]);
const smokeFinite = smokeOut.every(row => Array.from(row).every(Number.isFinite));

console.log(JSON.stringify({{
  concatDecoder: Array.from(concatModel.tensors.decoder.data),
  concatDecoderShape: concatModel.tensors.decoder.shape,
  concatEncoder: Array.from(concatModel.tensors.encoder.data),
  concatEncoderShape: concatModel.tensors.encoder.shape,
  concatLmHead: Array.from(concatModel.tensors.lm_head.data),
  concatN: concatModel.n,
  avgDecoder: Array.from(avgModel.tensors.decoder.data),
  avgDecoderShape: avgModel.tensors.decoder.shape,
  avgN: avgModel.n,
  smokeFinite,
  smokeShape: [smokeOut.length, smokeOut[0].length],
}}));
"""


@pytest.fixture(scope="module")
def merge_result():
    merge_uri = (WEB / "src/merge.js").as_uri()
    forward_uri = (WEB / "src/bdh_forward.js").as_uri()
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "merge_check.mjs"
        js.write_text(_NODE_SCRIPT.format(merge_uri=merge_uri, forward_uri=forward_uri))
        res = subprocess.run(["node", str(js)], capture_output=True, text=True, check=True)
        return json.loads(res.stdout.strip())


def test_merge_concat_decoder_is_plain_row_append(merge_result):
    # decoder: concat_axis 0 -> A's rows then B's rows, not interleaved.
    a_rows = [1, 2, 3, 4, 5, 6, 7, 8]
    b_rows = [101, 102, 103, 104, 105, 106, 107, 108]
    assert merge_result["concatDecoder"] == a_rows + b_rows
    assert merge_result["concatDecoderShape"] == [8, 2]


def test_merge_concat_encoder_interleaves_per_row_on_last_axis(merge_result):
    # encoder: concat_axis 2 (last axis) -> per D-row, A's n columns then B's n columns.
    # A naive whole-array append would give [row0_A, row1_A, row0_B, row1_B] instead —
    # this assertion distinguishes the two.
    expected = [
        10, 11, 12, 13, 110, 111, 112, 113,  # row i=0: A0..A3, B0..B3
        20, 21, 22, 23, 120, 121, 122, 123,  # row i=1: A0..A3, B0..B3
    ]
    naive_append = [10, 11, 12, 13, 20, 21, 22, 23, 110, 111, 112, 113, 120, 121, 122, 123]
    assert merge_result["concatEncoder"] == expected
    assert merge_result["concatEncoder"] != naive_append
    assert merge_result["concatEncoderShape"] == [1, 2, 8]


def test_merge_concat_averages_tensors_with_no_neuron_axis(merge_result):
    # lm_head has concat_axis: null -> averaged even though the merge mode is "concat".
    assert merge_result["concatLmHead"] == pytest.approx([52, 54, 56, 58])


def test_merge_average_decoder_is_elementwise_mean(merge_result):
    a_rows = [1, 2, 3, 4, 5, 6, 7, 8]
    b_rows = [101, 102, 103, 104, 105, 106, 107, 108]
    expected = [(x + y) / 2 for x, y in zip(a_rows, b_rows)]
    assert merge_result["avgDecoder"] == pytest.approx(expected)
    assert merge_result["avgDecoderShape"] == [4, 2]


def test_merge_n_doubles_under_concat_and_is_unchanged_under_average(merge_result):
    assert merge_result["concatN"] == 8   # 2 * a.n
    assert merge_result["avgN"] == 4      # a.n, unchanged


def test_merged_concat_model_produces_finite_logits(merge_result):
    # Smoke check only: a mis-interleaved merge can still produce finite numbers, so
    # this does not replace the layout assertions above — it just confirms the merged
    # model (n=8 after concat) is still runnable end to end through forward().
    assert merge_result["smokeFinite"] is True
    assert merge_result["smokeShape"] == [2, 2]  # 2 tokens, vocab_size=2
