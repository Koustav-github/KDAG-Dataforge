// Global reactive state for the model-surgery explainer.
//
// `theta` is the single θ control shared by every act: Act 1/2 react to
// whichever θ's parent weights are currently loaded, and Act 3's phase
// diagram marker tracks the same value. `mergeMode` is the concat/average
// toggle. `parentA`/`parentB` hold the two loaded Model objects for the
// current θ (set by App.svelte once loadModel() resolves); `mergedModel` is
// a *derived* store, so flipping `mergeMode` alone recomputes the merge
// synchronously from the already-loaded parent tensors — no network round
// trip, which is what keeps the toggle under a second.
import { writable, derived } from 'svelte/store';
import { mergeConcat, mergeAverage } from '../merge.js';

export const manifest = writable(null);

// String key into manifest.featured, e.g. "0.5". Only ever set to a θ that
// has weights on disk.
export const theta = writable('0.5');

export const mergeMode = writable('concat');

// Indices into the merged neuron axis to ablate. Unused until Task 14 —
// kept here so NeuronStrip's `highlight` prop already has a source.
export const ablated = writable([]);

export const parentA = writable(null);
export const parentB = writable(null);

// Guided-narrative progress: 1..4 while stepping through Acts 1-4, 'free'
// once the learner has seen all four and every control unlocks. Read by
// App.svelte to gate each section, and by Narrative.svelte to drive the
// stepper itself. Starts at 1 (not 0) so the page opens mid-Act-1, already
// showing a fused preset, rather than on a blank pre-narrative screen.
export const narrativeStep = writable(1);

export const mergedModel = derived(
  [parentA, parentB, mergeMode],
  ([$parentA, $parentB, $mergeMode]) => {
    if (!$parentA || !$parentB) return null;
    return $mergeMode === 'concat' ? mergeConcat($parentA, $parentB) : mergeAverage($parentA, $parentB);
  },
);
