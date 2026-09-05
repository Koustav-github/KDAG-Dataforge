import { forward } from '../bdh_forward.js';
import { EOS } from './tokens.js';

// Greedy-decodes up to `maxNew` tokens after `promptTokens`, stopping early
// on EOS. Sequences here are short (<=16), so recomputing forward() over
// the whole running sequence at each step is cheap and keeps this stateless
// with respect to bdh_forward.js's recurrent state.
export function greedyDecode(model, promptTokens, maxNew, { ablated = null } = {}) {
  const seq = promptTokens.slice();
  for (let step = 0; step < maxNew; step++) {
    const logits = forward(model, seq, { ablated });
    const last = logits[logits.length - 1];
    let best = 0;
    let bestVal = -Infinity;
    for (let v = 0; v < last.length; v++) {
      if (last[v] > bestVal) {
        bestVal = last[v];
        best = v;
      }
    }
    seq.push(best);
    if (best === EOS) break;
  }
  return seq.slice(promptTokens.length);
}
