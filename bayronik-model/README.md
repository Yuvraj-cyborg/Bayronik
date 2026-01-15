# bayronik-model

Neural network for baryonic feedback emulation. Maps gravity-only density fields to total matter fields.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[train]"
```

## Usage

```bash
# Download data
python download_data.py --dataset CV

# Train
python train.py --dataset CV --model unet --epochs 20
python train.py --dataset LH --model attention --mmap --epochs 50

# Export for Rust inference
python export.py --weights weights/best_attention_LH.pth --output weights/traced.pt --model attention
```

## Models

| Model | Params | Description |
|-------|--------|-------------|
| `unet` | 7.7M | Standard U-Net |
| `resunet` | 7.8M | Residual blocks |
| `attention` | 7.9M | Self-attention in bottleneck |

## Data

CAMELS IllustrisTNG maps from [Flatiron Institute](https://camels.readthedocs.io/).

- CV: 27 simulations (~140 MB)
- LH: 1000 simulations (~5 GB)
