# Bayronik Model

**SOTA Baryonic Feedback Emulator using Fourier Neural Operators**

This package provides neural operator architectures for solving the "baryonic bottleneck" in cosmology - learning the mapping from cheap gravity-only simulations to expensive hydrodynamic simulations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Fourier Neural Operators (FNO)**: State-of-the-art architecture for PDE-like mappings
- **U-FNO Hybrid**: Combines U-Net's local features with FNO's global spectral convolutions
- **Multi-scale Loss Functions**: Pixel + Power Spectrum + Higher-order Statistics
- **Conditional Models**: Support for physics parameters (redshift, feedback, cosmology)
- **ONNX Export**: Ready for WASM/browser deployment
- **CAMELS Integration**: Built for the CAMELS simulation dataset

## Installation

Using `uv` (recommended):

```bash
cd bayronik-model
uv venv
source .venv/bin/activate
uv sync

# With training dependencies
uv sync --extra train

# With export dependencies
uv sync --extra export

# Everything
uv sync --extra all
```

Using pip:

```bash
pip install -e ".[all]"
```

## Quick Start

### Training

```bash
# Download CAMELS data
uv run python -m bayronik_model.cli download --dataset CV

# Train U-FNO model (recommended)
uv run python train.py --model ufno --dataset CV --epochs 50

# Train with physics conditioning
uv run python train.py --model ufno_cond --dataset LH --conditional --epochs 100

# Train with wandb logging
uv run python train.py --model ufno --dataset LH --wandb --epochs 100
```

### Inference

```python
import torch
from bayronik_model import UFNO2d, get_model

# Load model
model = get_model("ufno", base_channels=32, modes=32, depth=4)
model.load_state_dict(torch.load("weights/best_ufno_CV.pth"))
model.eval()

# Run inference
dm_field = torch.randn(1, 1, 256, 256)  # Your gravity-only field
with torch.no_grad():
    hydro_field = model(dm_field)
```

### Export for Deployment

```python
from bayronik_model import export_onnx, export_torchscript

# Export to TorchScript (for Rust inference)
export_torchscript(model, "model.pt")

# Export to ONNX (for WASM/browser)
export_onnx(model, "model.onnx")
```

## Model Architectures

### Fourier Neural Operator (FNO)

```python
from bayronik_model import FNO2d

model = FNO2d(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes_x=32,           # Fourier modes in x
    modes_y=32,           # Fourier modes in y
    num_layers=4,
    dropout=0.0,
)
```

### U-FNO (Recommended)

```python
from bayronik_model import UFNO2d

model = UFNO2d(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    modes=32,
    depth=4,
    dropout=0.0,
)
```

### Conditional U-FNO

```python
from bayronik_model import UFNO2dConditional

model = UFNO2dConditional(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    modes=32,
    depth=4,
    num_conditions=6,  # z, A_AGN, A_SN, Omega_m, sigma_8, ...
)

# Forward with conditions
conditions = torch.tensor([[0.0, 1.0, 1.0, 0.3, 0.8, 0.05]])
output = model(input_dm, conditions)
```

## Loss Functions

The package includes physics-informed multi-scale losses:

```python
from bayronik_model import BaryonicEmulatorLoss

criterion = BaryonicEmulatorLoss(
    pixel_weight=1.0,      # MSE loss
    spectral_weight=0.1,   # Power spectrum loss
    stats_weight=0.1,      # Skewness, kurtosis
    gradient_weight=0.05,  # Edge preservation
    multiscale_weight=0.1, # Multi-resolution
)

losses = criterion(pred, target, input_dm)
# losses['total'], losses['pixel'], losses['spectral'], ...
```

## Dataset

```python
from bayronik_model import CAMELSDataset, create_dataloaders

# Simple usage
dataset = CAMELSDataset(
    data_dir="data",
    suite="IllustrisTNG",
    dataset_type="LH",  # or "CV"
    augment=True,
    return_params=True,  # Return physics parameters
)

# Create train/val loaders
train_loader, val_loader = create_dataloaders(
    data_dir="data",
    batch_size=8,
    dataset_type="LH",
    augment_train=True,
    return_params=True,
)
```

## Benchmarks

| Model | Parameters | Inference (CPU) | Inference (GPU) |
|-------|------------|-----------------|-----------------|
| UNet | 31M | 45ms | 3ms |
| FNO2d | 4.2M | 12ms | 1.5ms |
| UFNO2d | 2.1M | 18ms | 2ms |
| UFNO2d (cond) | 2.3M | 20ms | 2.2ms |

## Project Structure

```
bayronik-model/
├── src/bayronik_model/
│   ├── __init__.py      # Package exports
│   ├── model.py         # Traditional UNet architectures
│   ├── fno.py           # Fourier Neural Operators
│   ├── ufno.py          # U-FNO hybrid models
│   ├── losses.py        # Multi-scale loss functions
│   ├── dataset.py       # CAMELS dataset loader
│   ├── export.py        # TorchScript/ONNX export
│   └── cli.py           # Command-line interface
├── train.py             # Training script
├── pyproject.toml       # Package configuration
└── README.md
```

## Training on Cloud (A100)

```bash
# Setup on GCP with A100
gcloud compute ssh your-instance

# Clone and setup
git clone https://github.com/yourusername/bayronik.git
cd bayronik/bayronik-model

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv
source .venv/bin/activate
uv sync --extra train

# Download full LH dataset (~15GB)
uv run python -m bayronik_model.cli download --dataset LH

# Train with mixed precision
uv run python train.py \
    --model ufno \
    --dataset LH \
    --epochs 100 \
    --batch-size 32 \
    --amp \
    --wandb
```

## References

- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) - Li et al., 2020
- [U-FNO](https://www.sciencedirect.com/science/article/pii/S0309170822000562) - Wen et al., 2022
- [CAMELS](https://camels.readthedocs.io/) - Villaescusa-Navarro et al., 2022
- [BACCO](https://arxiv.org/abs/2011.15018) - Aricò et al., 2020

## License

MIT License

## Citation

```bibtex
@software{bayronik2025,
  author = {Yuvraj Biswal},
  title = {Bayronik: SOTA Baryonic Field Emulator},
  year = {2025},
  url = {https://github.com/yourusername/bayronik}
}
```
