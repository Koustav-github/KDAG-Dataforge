// Teacher-forced next-token cross-entropy, the same quantity
// model/src/bdh_surgery/train.py:evaluate computes in Python.
//
// Lives here because three call sites need it — the derivation panel, Act 4's
// ablation comparison, and the phase diagram's live marker — and a drifting
// copy in any one of them would silently disagree with the others.
import { forward } from '../bdh_forward.js';

export function sequenceLoss(model, tokenIds, ablated = null) {
  const input = tokenIds.slice(0, -1);
  const target = tokenIds.slice(1);
  const seq = forward(model, input, { ablated });
  let total = 0;
  for (let t = 0; t < seq.length; t++) {
    const logits = seq[t];
    let maxV = -Infinity;
    for (let v = 0; v < logits.length; v++) if (logits[v] > maxV) maxV = logits[v];
    let sumExp = 0;
    for (let v = 0; v < logits.length; v++) sumExp += Math.exp(logits[v] - maxV);
    total += Math.log(sumExp) + maxV - logits[target[t]]; // -log softmax[target]
  }
  return total / seq.length;
}

export function meanLoss(model, rows, ablated = null) {
  if (!model || !rows || !rows.length) return null;
  let total = 0;
  for (const row of rows) total += sequenceLoss(model, row, ablated);
  return total / rows.length;
}

// Merge damage, matching model/src/bdh_surgery/damage.py: each parent is
// scored on its OWN direction, and the two are reported separately as well as
// pooled — the per-direction split is the part a mean would hide.
export function measureDamage(merged, parentA, parentB, rowsA, rowsB, ablated = null) {
  const mA = meanLoss(merged, rowsA, ablated);
  const mB = meanLoss(merged, rowsB, ablated);
  const pA = meanLoss(parentA, rowsA, null);
  const pB = meanLoss(parentB, rowsB, null);
  if ([mA, mB, pA, pB].some((v) => v === null || !Number.isFinite(v))) return null;
  return {
    mergedA: mA, mergedB: mB, parentA: pA, parentB: pB,
    dA: mA - pA, dB: mB - pB, dMean: (mA - pA + (mB - pB)) / 2,
  };
}
