# model

Training and offline scientific validation pipeline (Python) for the Bayronik
baryonic field emulator. Production inference is served by the `server` crate.

## Setup

```bash
uv venv && uv sync
uv sync --extra train     # matplotlib, wandb, h5py
uv sync --extra dev       # pytest, ruff, mypy
```

## Training

```bash
python train.py --model ufno_cond --conditional --dataset LH \
    --epochs 50 --batch-size 16 --patience 15 --no-amp \
    --spectral-weight 0.5 --mass-weight 0.01 --verbose
```

Weights: `weights/best_{model}_{dataset}_{suite}.pth`.

## Scientific validation

```bash
make phase2   # validation.py -> registry CLI -> cargo test -p registry
```

Runs the LH + CV validation pipeline (Python), feeds the resulting
`reports/validation_report.json` into the `registry` CLI, which rewrites
`weights/model_registry.json` and gates it on the frozen thresholds.

## Layout

- `src/bayronik_model/ufno.py` — `UFNO2dConditional`
- `src/bayronik_model/dataset.py` — CAMELS dataset, conditioning expansion
- `src/bayronik_model/losses.py` — multi-scale + spectral loss
- `train.py` — training entry point
- `benchmarks/validation.py` — LH + CV scientific validation
- `tests/` — fast model-shape / behaviour checks

The Python module name (`bayronik_model`) is unchanged so existing imports keep
working; only the folder name is `model/`.
