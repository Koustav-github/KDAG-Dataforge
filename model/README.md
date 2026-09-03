# bdh-surgery (model)

Trains two toy BDH-GPU models on synthetic domains, merges them per the
concatenation rule in arXiv:2509.26507 §7.1, and exports weights + metadata to
`../web/public/data/` for the browser sandbox.

```
uv sync
uv run pytest
```

Pipeline: `domains.py` → `train.py` → `merge.py` → `export.py`. `bdh.py` holds
the BDH-GPU forward pass shared by training and merging.
