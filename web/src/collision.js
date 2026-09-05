// Collision neurons: units in the merged model that respond to both target
// languages. Ports model/src/bdh_surgery/ablate.py exactly — see that file
// for the rationale (a neuron inherited cleanly from one parent should be
// selective for that parent's direction; one that fires for both is where
// the two models compete for the same output channel).
import { neuronActivations } from './bdh_forward.js';

// probe may be a single token-id array (one sequence) or a batch of them
// (array of token-id arrays), matching how ablate.py's collision_scores
// is called with a batch of probe rows in the Python tests. Every row here
// has equal length, so averaging per-row means is the same as averaging
// over every (layer, position) pair combined, which is what
// activation_profiles(...).mean(axis=1) does in ablate.py.
function meanActivationProfile(model, probe) {
  const rows = Array.isArray(probe[0]) ? probe : [probe];
  const n = model.n;
  const sum = new Float64Array(n);
  for (const row of rows) {
    const act = neuronActivations(model, row);
    for (let i = 0; i < n; i++) sum[i] += act[i];
  }
  const out = new Float32Array(n);
  const inv = 1 / rows.length;
  for (let i = 0; i < n; i++) out[i] = sum[i] * inv;
  return out;
}

// a = mean activation profile of the merged model on probe_a, b = on
// probe_b. joint = min(a, b) (active on both directions), imbalance =
// |a-b| / (a+b+1e-8) (0 when equally active on both), score = joint * (1 -
// imbalance). High where a neuron is strongly AND equally active on both.
// Same computation as collisionScores, but keeps the intermediate terms so a
// derivation panel can show the formula being evaluated on a real neuron
// instead of asserting the result.
export function collisionBreakdown(model, probeA, probeB) {
  const a = meanActivationProfile(model, probeA);
  const b = meanActivationProfile(model, probeB);
  const n = model.n;
  const joint = new Float32Array(n);
  const imbalance = new Float32Array(n);
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const ai = a[i], bi = b[i];
    joint[i] = Math.min(ai, bi);
    imbalance[i] = Math.abs(ai - bi) / (ai + bi + 1e-8);
    scores[i] = joint[i] * (1 - imbalance[i]);
  }
  return { a, b, joint, imbalance, scores };
}

export function collisionScores(model, probeA, probeB) {
  // Delegates, so there is exactly one implementation of the score. Computing
  // it twice independently drifted by ~4e-9 on a few hundred neurons — harmless
  // for the ranking, but it meant the derivation panel and the heat-map could
  // in principle disagree about a value they both call "the score".
  return collisionBreakdown(model, probeA, probeB).scores;
}

// Indices of the k highest-scoring neurons, descending.
export function topK(scores, k) {
  const idx = Array.from({ length: scores.length }, (_, i) => i);
  idx.sort((x, y) => scores[y] - scores[x]);
  return idx.slice(0, k);
}

// mulberry32: a tiny, deterministic 32-bit PRNG. Not numpy's PCG64 (no
// requirement to match Python's bit stream, only to be reproducible from a
// seed within this JS port — see random_neurons in ablate.py).
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// k distinct indices in [0, nTotal), reproducible from `seed`. Size-matched
// control for the collision set — without this, "we ablated and it
// improved" proves nothing.
export function randomNeurons(nTotal, k, seed) {
  const rng = mulberry32(seed);
  const pool = Array.from({ length: nTotal }, (_, i) => i);
  const out = [];
  let remaining = nTotal;
  for (let i = 0; i < k; i++) {
    const j = Math.floor(rng() * remaining);
    out.push(pool[j]);
    pool[j] = pool[remaining - 1];
    remaining -= 1;
  }
  return out;
}
