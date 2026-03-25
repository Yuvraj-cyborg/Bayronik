# Bayronik

Field-level baryonic emulator for weak lensing cosmology.

Bayronik learns to map gravity-only dark matter density fields to total matter fields (dark matter + baryons), replacing months of hydrodynamic simulation with millisecond inference. It couples a Rust N-body simulator with a conditional U-FNO (Fourier Neural Operator) trained on the CAMELS dataset.

![license](https://img.shields.io/badge/license-MIT-blue.svg)
![rust](https://img.shields.io/badge/Rust-1.70%2B-%23dea584)
![python](https://img.shields.io/badge/Python-3.10%2B-%233776AB)

## Why This Exists

Modern weak lensing surveys (Euclid, LSST, Roman) need sub-percent accuracy on the matter power spectrum. Baryonic feedback (AGN jets, supernovae) redistributes matter on kpc-Mpc scales, biasing cosmological parameter inference by up to 10%. Full hydrodynamic simulations cost millions of CPU-hours. Bayronik does the same correction in ~5ms per map.

## Architecture

```
bayronik-core (Rust)           bayronik-model (Python)
Particle-Mesh N-body    -->    U-FNO + FiLM conditioning
Zel'dovich ICs                 Trained on CAMELS LH (15k maps)
CIC + FFT Poisson              Multi-scale loss (pixel + spectral + mass)
KDK integrator                 6 condition params: Om, s8, ASN1, AAGN1, ASN2, AAGN2
2D projection (256x256)        Streamlit dashboard with validation plots
```

The model is a U-Net enhanced Fourier Neural Operator (U-FNO) that combines:
- Spectral convolutions for global correlations (cosmic web, voids)
- U-Net encoder-decoder for multi-scale local features (halo profiles)
- FiLM conditioning for parameter-dependent predictions

## Quick Start

Prerequisites: Rust 1.70+, Python 3.10+, `uv` (recommended) or pip.

```bash
git clone https://github.com/yuvrajbiswal/bayronik.git
cd bayronik

# Download training/test data (~15 GB for LH)
make download-lh

# Launch the interactive dashboard
make demo
```

This opens the Streamlit app at http://localhost:8501 with:
- CAMELS data viewer with inference and ground truth comparison
- N-body simulator integration (generates custom DM maps)
- Parameter sweep for exploring feedback sensitivity
- Validation plots: P(k), baryon suppression S(k), 1-point PDF

### Other Commands

```bash
make help              # Show all targets
make train             # Train conditional U-FNO on LH data
make build-nbody       # Build N-body simulator binary
make infer             # Run Rust TUI (needs libtorch)
make server            # FastAPI inference server
```

## Project Layout

```
bayronik/
  bayronik-core/       Rust PM N-body: CIC, FFT Poisson, KDK, Zel'dovich ICs
  bayronik-model/      Python: U-FNO, training, losses, dataset, server, webapp
  bayronik-infer/      Rust TUI: tch-rs inference with terminal heatmaps
  bayronik-web/        Rust egui frontend (experimental)
  Makefile             Build, train, demo orchestration
```

### bayronik-model/src/bayronik_model/

| File | Purpose |
|------|---------|
| `ufno.py` | UFNO2d, UFNO2dConditional, AttentionUFNO2d |
| `fno.py` | FNO2d, SpectralConv2d, FNO2dConditional |
| `model.py` | UNet, ResUNet, AttentionUNet (baseline) |
| `losses.py` | BaryonicEmulatorLoss: pixel + spectral + stats + gradient + mass conservation |
| `dataset.py` | CAMELSDataset with multi-projection param expansion |
| `export.py` | TorchScript and ONNX export |

## Data

Training uses the [CAMELS Multifield Dataset](https://camels.readthedocs.io/) (CMD), specifically 2D projected maps from IllustrisTNG:

| File | Shape | Description |
|------|-------|-------------|
| `Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy` | (15000, 256, 256) | Dark matter density (input) |
| `Maps_Mtot_IllustrisTNG_LH_z=0.00.npy` | (15000, 256, 256) | Total matter density (target) |
| `params_LH_IllustrisTNG.txt` | (1000, 6) | Om, s8, ASN1, AAGN1, ASN2, AAGN2 per simulation |

1000 simulations x 15 projections = 15,000 training pairs. The parameter file has one row per simulation; the dataset loader repeats each row 15 times to match maps.

## Training

```bash
# On a GPU machine (GCP L4 recommended):
python train.py --model ufno_cond --conditional --dataset LH \
    --epochs 100 --patience 20 --no-amp \
    --spectral-weight 0.5 --mass-weight 0.01 --verbose
```

The loss function combines:
- **Pixel MSE** (weight=1.0): field-level accuracy
- **Power spectrum** (weight=0.5): statistical accuracy across scales
- **Field statistics** (weight=0.1): mean, variance, skewness, kurtosis
- **Gradient** (weight=0.05): edge/structure preservation
- **Multi-scale** (weight=0.1): features at all resolutions
- **Mass conservation** (weight=0.01): physical constraint on total mass

AMP is disabled because FNO's complex FFT operations don't support half precision on CUDA.

## Validation Metrics

The Streamlit dashboard computes:

- **P(k)**: 2D isotropic power spectrum comparison (input, prediction, ground truth)
- **S(k) = P_total(k) / P_DM(k)**: Baryon suppression ratio (should match ground truth curve)
- **1-point PDF**: Pixel value distribution in log-density space
- **Log MSE/MAE**: Error metrics in log space (appropriate for high dynamic range fields)
- **Pixel relative error**: Per-pixel absolute error normalized by ground truth

## N-Body Simulator

`bayronik-core` implements a Particle-Mesh N-body code in Rust:

1. **Initial conditions**: Zel'dovich approximation from Gaussian random field
2. **Mass assignment**: Cloud-in-Cell (CIC) interpolation
3. **Gravity**: FFT Poisson solver in k-space
4. **Forces**: Finite differences on potential
5. **Integration**: Symplectic Kick-Drift-Kick with periodic boundaries
6. **Projection**: 3D to 2D surface density via CIC

The Streamlit app calls this binary to generate custom DM maps on-the-fly, which are then fed through the emulator.

## References

- CAMELS: Villaescusa-Navarro et al., 2021 ([arXiv:2109.10915](https://arxiv.org/abs/2109.10915))
- U-FNO: Wen et al., 2022 ([doi:10.1016/j.advwatres.2022.104180](https://doi.org/10.1016/j.advwatres.2022.104180))
- FNO: Li et al., 2021 ([arXiv:2010.08895](https://arxiv.org/abs/2010.08895))
- Baryonic effects on lensing: Schneider & Teyssier, 2015 ([arXiv:1510.06034](https://arxiv.org/abs/1510.06034))
- CAMELS Multifield Dataset: Villaescusa-Navarro et al., 2022 ([arXiv:2109.10915](https://arxiv.org/abs/2109.10915))
- IllustrisTNG: [tng-project.org](https://www.tng-project.org/)

## License

MIT

## Author

Yuvraj Biswal -- yuvrajbiswalofficial@gmail.com
