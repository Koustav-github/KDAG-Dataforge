// Device capability hints, in one place so the tuning is auditable.
//
// IMPORTANT: nothing here changes a computed number. Mobile runs the exact same
// sequence counts, row counts and random draws as desktop, so a phone and a
// laptop report identical losses and damage. These knobs only change *how the
// work is sliced* and *how many pixels the canvases fill* — never what is
// measured. A device-dependent statistic would be a correctness bug in an
// artifact whose whole point is honest measurement.

// Coarse pointer (touch) is a better proxy for "phone/tablet" than screen width,
// which a desktop user can produce just by narrowing a window.
export function isCoarsePointer() {
  return typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
}

// hardwareConcurrency is a rough stand-in for CPU budget; it is absent on some
// browsers, so treat unknown as "not constrained" rather than guessing low.
export function isLowCore() {
  const n = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : undefined;
  return typeof n === 'number' && n > 0 && n <= 4;
}

export function isConstrained() {
  return isCoarsePointer() || isLowCore();
}

// How many sequences to run between yields to the browser.
//
// A forward pass over one sequence costs ~50 ms on a laptop and 3-5x that on a
// phone. Yielding every 2 sequences therefore blocks a phone for up to ~500 ms
// at a stretch, which reads as a frozen page. One sequence per slice keeps the
// longest block to a single forward pass; total work is unchanged.
export function lossChunk() {
  return isConstrained() ? 1 : 2;
}

// Canvas backing-store scale. Uncapped devicePixelRatio is the classic mobile
// performance trap: a DPR-3 phone fills 9x the pixels of a DPR-1 screen for a
// difference almost nobody can see on a heatmap of flat-shaded cells. Cap at 2.
export function canvasScale() {
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  return Math.min(dpr, 2);
}

// Slider debounce. A touch drag emits far more intermediate values than a mouse
// drag, and each committed value can queue a full recompute, so hold longer
// before committing on touch devices.
export function commitDelayMs() {
  return isConstrained() ? 260 : 140;
}
