// BDH-GPU forward pass (Eq. 8, arXiv:2509.26507) for the browser — four matmuls,
// no runtime dependency. Reads weights exported by model/src/bdh_surgery/export.py.
// TODO: implement once model/export.py's output schema is frozen.

export function forward(weights, input) {
  throw new Error("not implemented");
}
