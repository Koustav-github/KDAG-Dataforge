// Vocabulary layout (fixed by the Python data generator):
//   PAD=0, pivot concepts 1..24, target surface tokens 25..72,
//   BOS_A=73, BOS_B=74, SEP=75, EOS=76.
export const PAD = 0;
export const BOS_A = 73;
export const BOS_B = 74;
export const SEP = 75;
export const EOS = 76;

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

export function labelToken(id, lexicon) {
  if (id === PAD) return '·'; // ·
  if (id === BOS_A) return '⟨A⟩'; // ⟨A⟩
  if (id === BOS_B) return '⟨B⟩'; // ⟨B⟩
  if (id === SEP) return '|';
  if (id === EOS) return '∎'; // ∎
  if (id >= 1 && id <= 24) return `p${id}`;
  if (lexicon) {
    if (lexicon.a.has(id)) return `a${lexicon.a.get(id)}`;
    if (lexicon.b.has(id)) return `b${lexicon.b.get(id)}`;
  }
  return `t${id}`;
}

export function labelSequence(ids, lexicon) {
  return ids.map((id) => labelToken(id, lexicon));
}

// Splits a probe row [BOS, ...concepts, SEP, ...target, EOS, 0, 0, ...]
// into the prompt (through SEP) and the oracle continuation (through EOS,
// padding stripped).
export function splitProbe(seq) {
  const sepIdx = seq.indexOf(SEP);
  const eosIdx = seq.indexOf(EOS);
  const prompt = sepIdx >= 0 ? seq.slice(0, sepIdx + 1) : seq.slice();
  const target = sepIdx >= 0 && eosIdx >= 0 ? seq.slice(sepIdx + 1, eosIdx + 1) : [];
  return { prompt, target };
}
