# Bayronik

Field-level baryonic emulator for weak lensing cosmology.

Bayronik learns to map gravity-only dark matter density fields to total matter fields (dark matter + baryons), replacing months of hydrodynamic simulation with millisecond inference. It couples a Rust N-body simulator with a conditional U-FNO (Fourier Neural Operator) trained on the CAMELS dataset.

![license](https://img.shields.io/badge/license-MIT-blue.svg)
![rust](https://img.shields.io/badge/Rust-1.92%2B-%23dea584)
![python](https://img.shields.io/badge/Python-3.10%2B-%233776AB)

## Why This Exists

Modern weak lensing surveys (Euclid, LSST, Roman) need sub-percent accuracy on the matter power spectrum. Baryonic feedback (AGN jets, supernovae) redistributes matter on kpc-Mpc scales, biasing cosmological parameter inference by up to 10%. Full hydrodynamic simulations cost millions of CPU-hours. Bayronik does the same correction in ~5ms per map.

## Architecture

```
engine      (Rust)         model     (Python — train only)
Particle-Mesh N-body        -->   U-FNO + FiLM conditioning
Zel'dovich ICs                    Trained on CAMELS LH (15k maps)
CIC + FFT Poisson                 Multi-scale loss (pixel + spectral + mass)
KDK integrator                    6 conditions: Ωm, σ8, ASN1, AAGN1, ASN2, AAGN2

client       (Rust/WASM)    server    (Rust + tch + axum)
egui desktop + browser app        Pure-Rust HTTP inference backend
Client-side N-body via WASM       Loads TorchScript .pt directly, no Python
P(k), S(k), PDF analysis          CORS-enabled for WASM frontend

registry  (Rust)         infer     (Rust)
Model card / frozen thresholds    Local TUI inference (libtorch)
+ regression tests, no libtorch   tch-rs + ratatui
```

The model is a U-Net enhanced Fourier Neural Operator (U-FNO) that combines:
- Spectral convolutions for global correlations (cosmic web, voids)
- U-Net encoder-decoder for multi-scale local features (halo profiles)
- FiLM conditioning for parameter-dependent predictions

The frontend runs as a native desktop app or compiles to WebAssembly for the browser. The N-body simulator runs entirely client-side in WASM — no server needed for dark matter map generation.

## Quick Start

Prerequisites: Rust 1.85+, Python 3.10+, `uv` (recommended) or pip.

```bash
git clone https://github.com/cosmexus/bayronik.git
cd bayronik

make download-lh       # CAMELS LH maps + params (~15 GB)
make server            # boot inference HTTP server on :8000
make client            # native egui app   (or: make wasm for browser on :8080)
```

The client has four tabs:
- **N-Body Simulator** — generate DM maps client-side via WASM, run through emulator
- **CAMELS Data** — browse samples, compare inference to ground truth
- **Parameter Sweep** — vary one parameter, observe baryonic effects
- **About** — project info and parameter definitions

### Other targets

```bash
make help              # all targets
make train             # train conditional U-FNO on LH
make validate          # LH + CV scientific validation -> reports/
make phase2            # validate -> registry rebuild -> regression tests
make test              # engine + registry + fast model tests
```

## Project Layout

```
bayronik/
  engine/       Rust PM N-body: CIC, FFT Poisson, KDK, Zel'dovich ICs
  model/      Python: U-FNO architecture, training, losses, dataset, validation
  server/     Rust: axum + tch HTTP inference backend (production path)
  registry/   Rust: model registry types + builder + regression tests (no libtorch)
  infer/      Rust TUI: tch-rs inference with terminal heatmaps
  client/       Rust egui frontend (desktop + WASM), client-side analysis
  Makefile      One Makefile, prod targets only
  flake.nix     Nix dev shell
  docs/         Model card, deployment, brand, research ideas
```

### model/src/bayronik_model/

| File | Purpose |
|------|---------|
| `ufno.py` | UFNO2d, UFNO2dConditional, AttentionUFNO2d |
| `fno.py` | FNO2d, SpectralConv2d, FNO2dConditional |
| `model.py` | UNet, ResUNet, AttentionUNet (baseline) |
| `losses.py` | BaryonicEmulatorLoss: pixel + spectral + stats + gradient + mass conservation |
| `dataset.py` | CAMELSDataset with multi-projection param expansion |
| `export.py` | TorchScript and ONNX export |

### client/src/

| File | Purpose |
|------|---------|
| `app.rs` | Main app: 4-tab UI, server communication, plotting |
| `visualization.rs` | Colormaps (Inferno, Viridis, DarkDiverging, etc.), heatmap rendering |
| `analysis.rs` | Client-side power spectrum, baryon suppression, PDF (pure Rust) |
| `lib.rs` | WASM entry point |

## Data

Training uses the [CAMELS Multifield Dataset](https://camels.readthedocs.io/) (CMD), specifically 2D projected maps from IllustrisTNG:

| File | Shape | Description |
|------|-------|-------------|
| `Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy` | (15000, 256, 256) | Dark matter density (input) |
| `Maps_Mtot_IllustrisTNG_LH_z=0.00.npy` | (15000, 256, 256) | Total matter density (target) |
| `params_LH_IllustrisTNG.txt` | (1000, 6) | Ωm, σ8, ASN1, AAGN1, ASN2, AAGN2 per simulation |

1000 simulations x 15 projections = 15,000 training pairs. The parameter file has one row per simulation; the dataset loader repeats each row 15 times to match maps.

## Training

```bash
# On a GPU machine (H100/A100 recommended):
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

The frontend computes:

- **P(k)**: 2D isotropic power spectrum comparison (input, prediction, ground truth)
- **S(k) = P_total(k) / P_DM(k)**: Baryon suppression ratio
- **1-point PDF**: Pixel value distribution in log-density space
- **Log MSE/MAE**: Error metrics in log space (appropriate for high dynamic range fields)

## N-Body Simulator

`engine` implements a cosmological Particle-Mesh N-body code in Rust (flat LCDM, h-units: Mpc/h, Msun/h, H0 = 1):

1. **Cosmology**: E(a), exact linear growth D(a) and f(a), Eisenstein-Hu (1998) linear power spectrum normalized to sigma8
2. **Initial conditions**: Gaussian random field with the linear P(k) at z=49, Zel'dovich displacements and growing-mode momenta p = a²Ef·psi
3. **Mass assignment**: Cloud-in-Cell (CIC), particle mass = Ωm ρ_crit V / N
4. **Gravity**: FFT Poisson solver with the physical prefactor (3/2) Ωm/a
5. **Integration**: Symplectic KDK leapfrog in scale factor with exact kick (∫da/aE) and drift (∫da/a³E) factors
6. **Projection**: 2D surface density in (Msun/h)/(Mpc/h)², CAMELS map convention, with configurable slab depth

The simulator compiles to WebAssembly and runs entirely in the browser — no server round-trip needed for dark matter map generation. Because the PM mesh resolves fewer nonlinear scales than the 256³-particle CAMELS runs, the client applies an affine log-space calibration using training-set statistics served by the backend (`GET /stats`).

## Deployment

- **Frontend**: Static WASM site on Vercel/Netlify (build with `make wasm`)
- **Backend**: `server` (Rust, tch + axum) on Modal.com / Fly.io / Railway / AWS / Azure
- **Desktop**: Native app via `cargo run --release -p client`

See `docs/DEPLOYMENT.md` for the full operational plan and `docs/MODEL_CARD.md` for
the frozen scientific metrics enforced by `cargo test -p registry`.

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
