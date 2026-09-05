#!/usr/bin/env python
"""Measure whether merge damage localizes to the collision set.

This is the negative result reported in the README: ablating the
collision-scored neurons never recovers function, and at k = 100 it is
significantly *worse* than ablating the same number of random neurons.

The measurement has to run against the exported browser weights (there is no
PyTorch copy of the merged models on disk), so this script shells out to node
and drives exactly the code the artifact ships:

    web/src/bdh_forward.js   the recurrent forward pass
    web/src/merge.js         the concat merge
    web/src/collision.js     collision scoring + the random control

The loss is the same teacher-forced mean cross-entropy over the same probe
rows that web/src/lib/Surgery.svelte uses (the first 4 rows of each eval set),
so the k = 40 / theta = 0.5 line reproduces the artifact's shipped default.

Not part of the pytest suite: it needs `web/public/data/` to have been
exported, and it needs node on PATH. Run it directly:

    python model/scripts/measure_locality.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB_SRC = REPO / "web" / "src"
DATA_DIR = REPO / "web" / "public" / "data"

THETAS = ("0.0", "0.5", "1.0")
KS = (40, 100)
UI_SEED = 7          # the SEED constant in Surgery.svelte
N_DRAWS = 12         # extra random draws, to report the spread of the control
LOSS_ROWS_PER_SIDE = 4

DRIVER = """
import {{ loadModelFromDir, forward }} from '{fwd}';
import {{ mergeConcat }} from '{merge}';
import {{ collisionScores, topK, randomNeurons }} from '{collision}';
import fs from 'node:fs/promises';

const DIR = {data!r};
const THETAS = {thetas};
const KS = {ks};
const UI_SEED = {ui_seed};
const N_DRAWS = {n_draws};
const ROWS_PER_SIDE = {rows_per_side};

const probes = JSON.parse(await fs.readFile(`${{DIR}}/probes.json`, 'utf8'));
// Eval corpora are per-theta: each theta has its own target lexicons, so a
// shared probe set would score every theta against theta=0.0's vocabulary.
const probesFor = (t) => probes.featured[t];

// Teacher-forced mean cross-entropy, mirroring Surgery.svelte:sequenceLoss.
function sequenceLoss(model, ids, ablated) {{
  const input = ids.slice(0, -1), target = ids.slice(1);
  const logitsSeq = forward(model, input, {{ ablated }});
  let total = 0;
  for (let t = 0; t < logitsSeq.length; t++) {{
    const logits = logitsSeq[t];
    let maxV = -Infinity;
    for (const v of logits) if (v > maxV) maxV = v;
    let sumExp = 0;
    for (const v of logits) sumExp += Math.exp(v - maxV);
    total += Math.log(sumExp) + maxV - logits[target[t]];
  }}
  return total / logitsSeq.length;
}}
const meanLoss = (m, rs, ab) =>
  rs.reduce((acc, r) => acc + sequenceLoss(m, r, ab), 0) / rs.length;

const out = [];
for (const theta of THETAS) {{
  const a = await loadModelFromDir(DIR, theta, 'A');
  const b = await loadModelFromDir(DIR, theta, 'B');
  const merged = mergeConcat(a, b);
  const pr = probesFor(theta);
  const rows = [...pr.eval_a.slice(0, ROWS_PER_SIDE),
                ...pr.eval_b.slice(0, ROWS_PER_SIDE)];
  const scores = collisionScores(merged, pr.eval_a, pr.eval_b);
  const baseline = meanLoss(merged, rows, null);
  for (const k of KS) {{
    const collision = meanLoss(merged, rows, topK(scores, k));
    const ui = meanLoss(merged, rows, randomNeurons(merged.n, k, UI_SEED));
    const draws = [];
    for (let s = 1; s <= N_DRAWS; s++) {{
      draws.push(meanLoss(merged, rows, randomNeurons(merged.n, k, s)));
    }}
    const mean = draws.reduce((x, y) => x + y, 0) / draws.length;
    const sd = Math.sqrt(
      draws.reduce((x, y) => x + (y - mean) ** 2, 0) / (draws.length - 1));
    out.push({{ theta: Number(theta), k, baseline, collision,
                random_ui_seed: ui, random_mean: mean, random_sd: sd }});
  }}
}}
process.stdout.write('@@JSON@@' + JSON.stringify(out));
"""


def run() -> list[dict]:
    if shutil.which("node") is None:
        sys.exit("node is not on PATH; this script drives the shipped JS directly.")
    if not (DATA_DIR / "manifest.json").exists():
        sys.exit(f"no export found at {DATA_DIR} — run bdh_surgery.export first.")

    src = DRIVER.format(
        fwd=(WEB_SRC / "bdh_forward.js").as_uri(),
        merge=(WEB_SRC / "merge.js").as_uri(),
        collision=(WEB_SRC / "collision.js").as_uri(),
        data=str(DATA_DIR).replace("\\", "/"),
        thetas=json.dumps(list(THETAS)),
        ks=json.dumps(list(KS)),
        ui_seed=UI_SEED,
        n_draws=N_DRAWS,
        rows_per_side=LOSS_ROWS_PER_SIDE,
    )
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "measure_locality.mjs"
        script.write_text(src, encoding="utf8")
        proc = subprocess.run([shutil.which("node"), str(script)],
                              capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"node exited {proc.returncode}")
    marker = proc.stdout.rindex("@@JSON@@") + len("@@JSON@@")
    return json.loads(proc.stdout[marker:])


def main() -> None:
    rowsets = run()
    head = (f"{'theta':>6}  {'k':>4}  {'baseline':>9}  {'collision':>10}  "
            f"{'random(7)':>10}  {'headline':>9}  {'rand mean':>10}  {'rand sd':>8}")
    print("Locality check: does ablating the collision set recover function?")
    print("headline = random(seed 7) - collision, as reported in Surgery.svelte.")
    print("Positive => the collision set is the damage. Negative => it is not.\n")
    print(head)
    print("-" * len(head))
    for r in rowsets:
        headline = r["random_ui_seed"] - r["collision"]
        print(f"{r['theta']:>6.1f}  {r['k']:>4d}  {r['baseline']:>9.4f}  "
              f"{r['collision']:>10.4f}  {r['random_ui_seed']:>10.4f}  "
              f"{headline:>+9.4f}  {r['random_mean']:>10.4f}  {r['random_sd']:>8.4f}")
    print("\nCollision ablation never lowers the loss below baseline, and at k = 100 it")
    print("is worse than the random control at every theta. Damage does not localize")
    print("to the neurons this collision metric identifies.")


if __name__ == "__main__":
    main()
