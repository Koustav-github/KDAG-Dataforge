// Off-the-critical-path compute for the heavy panels.
//
// A forward pass over one 15-token sequence at n=2048, L=4 costs ~50 ms in
// plain JS. The panels need dozens of them per θ change, which — run straight
// through — is several seconds of blocked main thread: dead controls, no
// scrolling, no paint. Two things fix that:
//
//   1. yield to the browser between sequences, so it can paint and handle input
//   2. cache by an exact key, so revisiting a θ costs nothing
//
// Nothing here changes a number. Same definitions as model/src/bdh_surgery,
// just scheduled so the page stays alive while they run.
import { sequenceLoss } from './loss.js';

const nextFrame = () =>
  new Promise((r) =>
    typeof requestAnimationFrame === 'function' ? requestAnimationFrame(() => r()) : setTimeout(r, 0),
  );

// Run `fn` over each row, surrendering the thread every `chunk` rows.
async function mapYielding(rows, fn, chunk = 2) {
  const out = [];
  for (let i = 0; i < rows.length; i++) {
    out.push(fn(rows[i], i));
    if ((i + 1) % chunk === 0) await nextFrame();
  }
  return out;
}

export async function meanLossAsync(model, rows, ablated = null) {
  if (!model || !rows || !rows.length) return null;
  const each = await mapYielding(rows, (r) => sequenceLoss(model, r, ablated));
  return each.reduce((a, b) => a + b, 0) / each.length;
}

// Merge damage, matching model/src/bdh_surgery/damage.py — each parent scored
// only on the direction it was trained for.
export async function measureDamageAsync(merged, parentA, parentB, rowsA, rowsB, ablated = null) {
  const mA = await meanLossAsync(merged, rowsA, ablated);
  const mB = await meanLossAsync(merged, rowsB, ablated);
  const pA = await meanLossAsync(parentA, rowsA, null);
  const pB = await meanLossAsync(parentB, rowsB, null);
  if ([mA, mB, pA, pB].some((v) => v === null || !Number.isFinite(v))) return null;
  return {
    mergedA: mA, mergedB: mB, parentA: pA, parentB: pB,
    dA: mA - pA, dB: mB - pB, dMean: (mA - pA + (mB - pB)) / 2,
  };
}

// Damage is deterministic given (θ, merge rule, ablation set), so it is worth
// remembering: dragging back to a θ you have already visited is then free.
const damageCache = new Map();

export function damageKey(dataset, theta, mergeMode, ablated) {
  // `dataset` must be first-class here: baseline and large_vocab both produce
  // the string key "0.5|concat|none" from (theta, mergeMode, ablated) alone,
  // so two different models' damage would collide in the cache and a dataset
  // switch could silently show the PREVIOUS dataset's cached number.
  const abl = ablated && ablated.length ? `${ablated.length}:${ablated[0]}` : 'none';
  return `${dataset}|${theta}|${mergeMode}|${abl}`;
}

export async function cachedDamage(key, compute) {
  if (damageCache.has(key)) return damageCache.get(key);
  const promise = compute();
  damageCache.set(key, promise);   // store the promise so concurrent callers share one run
  try {
    return await promise;
  } catch (err) {
    damageCache.delete(key);
    throw err;
  }
}

export function clearDamageCache() {
  damageCache.clear();
}
