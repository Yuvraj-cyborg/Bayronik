# bayronik-model

Python package for training and serving the Bayronik baryonic field emulator.

## Setup

```bash
uv venv && uv sync
uv sync --extra train
uv sync --extra demo
```

## Training

```bash
python train.py --model ufno_cond --conditional --dataset LH \
    --epochs 50 --batch-size 16 --patience 15 --no-amp \
    --spectral-weight 0.5 --mass-weight 0.01 --verbose
```

Weights: `weights/best_{model}_{dataset}_{suite}.pth`

## Dashboard

```bash
make demo
```

## Inference server

```bash
uv run --extra server uvicorn server:app --host 0.0.0.0 --port 8000
```

## Layout

- `src/bayronik_model/ufno.py` — UFNO2dConditional
- `src/bayronik_model/dataset.py` — CAMELSDataset, param expansion for multi-projection maps
- `src/bayronik_model/losses.py` — BaryonicEmulatorLoss
- `train.py`, `webapp.py`, `server.py`

## Data

Place CAMELS maps under `data/` (see root README). Download: `make download-lh`.
