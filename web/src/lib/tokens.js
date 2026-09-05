// Vocabulary layout for the 24-concept datasets (baseline, long_phrase):
//   PAD=0, pivot concepts 1..24, target surface tokens 25..72,
//   BOS_A=73, BOS_B=74, SEP=75, EOS=76.
//
// large_vocab (40 concepts) does NOT use these — its layout is pivot 1..40,
// target 41..120, specials 121..124 (see model/src/bdh_surgery/domains.py's
// layout()). Every function below takes a `layout` object so the caller
// supplies the right one; these constants are only the default, used when a
// caller doesn't have a dataset-specific layout yet.
export const PAD = 0;
export const BOS_A = 73;
export const BOS_B = 74;
export const SEP = 75;
export const EOS = 76;

const DEFAULT_LAYOUT = { nConcepts: 24, pivotBase: 1, bosA: BOS_A, bosB: BOS_B, sep: SEP, eos: EOS };

// Mirrors model/src/bdh_surgery/domains.py:layout() exactly — same arithmetic,
// so a dataset's n_concepts alone is enough to derive where its specials and
// pivot range actually live.
export function layoutFor(nConcepts) {
  const pivotBase = 1;
  const targetBase = pivotBase + nConcepts;
  const bosA = targetBase + 2 * nConcepts;
  return { nConcepts, pivotBase, bosA, bosB: bosA + 1, sep: bosA + 2, eos: bosA + 3 };
}

// Builds surface-token-id -> label lookups from a θ's lex_a/lex_b (concept
// index -> that language's surface token id), so a raw id like 42 renders
// as "a7" or "b7" — which language produced it, and which pivot concept.
export function buildLexicon(featuredTheta) {
  const a = new Map();
  const b = new Map();
  (featuredTheta.lex_a || []).forEach((surfaceId, concept) => a.set(surfaceId, concept));
  (featuredTheta.lex_b || []).forEach((surfaceId, concept) => b.set(surfaceId, concept));
  return { a, b };
}

export function labelToken(id, lexicon, layout = DEFAULT_LAYOUT) {
  if (id === PAD) return '·'; // ·
  if (id === layout.bosA) return '⟨A⟩'; // ⟨A⟩
  if (id === layout.bosB) return '⟨B⟩'; // ⟨B⟩
  if (id === layout.sep) return '|';
  if (id === layout.eos) return '∎'; // ∎
  if (id >= layout.pivotBase && id < layout.pivotBase + layout.nConcepts) return `p${id}`;
  if (lexicon) {
    if (lexicon.a.has(id)) return `a${lexicon.a.get(id)}`;
    if (lexicon.b.has(id)) return `b${lexicon.b.get(id)}`;
  }
  return `t${id}`;
}

export function labelSequence(ids, lexicon, layout = DEFAULT_LAYOUT) {
  return ids.map((id) => labelToken(id, lexicon, layout));
}

// Splits a probe row [BOS, ...concepts, SEP, ...target, EOS, 0, 0, ...]
// into the prompt (through SEP) and the oracle continuation (through EOS,
// padding stripped).
export function splitProbe(seq, layout = DEFAULT_LAYOUT) {
  const sepIdx = seq.indexOf(layout.sep);
  const eosIdx = seq.indexOf(layout.eos);
  const prompt = sepIdx >= 0 ? seq.slice(0, sepIdx + 1) : seq.slice();
  const target = sepIdx >= 0 && eosIdx >= 0 ? seq.slice(sepIdx + 1, eosIdx + 1) : [];
  return { prompt, target };
}
