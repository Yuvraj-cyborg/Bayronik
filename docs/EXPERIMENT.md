# Bayronik: Architecture and Experiment Log

## What Bayronik Is

A field-level baryonic emulator for weak lensing cosmology. It takes a gravity-only
dark matter density map and predicts what that map would look like with full baryonic
physics (gas cooling, star formation, AGN/SN feedback) -- replacing months of
hydrodynamic simulation with millisecond inference.

## Current Architecture

```
                         ┌─────────────────────────────┐
                         │     bayronik-core (Rust)     │
                         │  Particle-Mesh N-body sim    │
                         │  Zel'dovich ICs → KDK → CIC │
                         │  Output: 256x256 DM map      │
                         └──────────┬──────────────────┘
                                    │ .npy
                                    v
┌──────────────────┐     ┌─────────────────────────────┐
│  CAMELS LH Data  │────>│    bayronik-model (Python)   │
│  15000 maps      │     │                              │
│  1000 sims x 15  │     │  U-FNO + FiLM conditioning   │
│  6 params each   │     │  Multi-scale loss function   │
└──────────────────┘     │  Streamlit dashboard         │
                         └──────────┬──────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    v               v               v
             ┌───────────┐  ┌────────────┐  ┌────────────┐
             │ webapp.py  │  │ server.py  │  │ TUI (Rust) │
             │ Streamlit  │  │ FastAPI    │  │ tch-rs     │
             └───────────┘  └────────────┘  └────────────┘
```

## Component Details

### 1. bayronik-core (Rust N-body engine)

Purpose: Generate synthetic dark matter density maps from first principles.
This lets users create custom inputs for the emulator without needing CAMELS data.

Implementation:
- Particle-Mesh (PM) scheme with Cloud-in-Cell (CIC) mass assignment
- FFT-based Poisson solver for gravity
- Kick-Drift-Kick symplectic leapfrog integrator
- Zel'dovich Approximation for initial conditions
- 3D -> 2D projection via CIC along line of sight
- Output: flat 256x256 surface density map as .npy

Fixed issues (March 2026):
- [FIXED] Poisson solver now uses batched 1D FFTs along each axis for proper 3D DFT
  (was: single 1D FFT of length N^3, which produced meaningless potential)
- [FIXED] KDK integration now recomputes density, potential, and forces after the
  drift step before the second half-kick (was: reusing stale forces, first-order error)
- [FIXED] Power spectrum scaling now correctly applies amplitude ~ k^(-0.75) for
  P(k) ~ k^(-1.5) (was: amplitude ~ k^(-1.5) giving P(k) ~ k^(-3))
- [FIXED] ZA displacement sign corrected (i/k^2 instead of -i/k^2)
- [FIXED] Deterministic RNG seed for reproducible initial conditions

Remaining limitations:
- Displacement and velocity scales are ad-hoc constants, not tied to
  cosmological parameters (Omega_m, sigma_8, growth factor D(z))
- No cosmological expansion (static Newtonian, not comoving coordinates)
- Single-threaded (no rayon parallelism)
- f32 precision throughout

With the FFT and KDK fixes, the engine now produces genuine gravitational
collapse with realistic overdensities and voids. Maps are suitable as demo
inputs and show qualitatively correct cosmic web structure.

### 2. bayronik-model (Python ML pipeline)

#### Model: UFNO2dConditional
- U-Net enhanced Fourier Neural Operator
- Encoder: SpectralConv2d + Conv2d blocks with downsampling
- Decoder: ConvTranspose2d upsampling with skip connections
- FiLM conditioning: 6 physics parameters modulate skip connections
  via per-layer (gamma, beta) projections from a shared MLP embedding
- Parameters: ~5-7M trainable params
- Depth 4, base_channels 32, modes 32

Channel flow (depth=4, base=32):
  Input 1 -> Lift 32 -> Enc[64,128,256,256] -> Bottleneck 256
  -> Dec[128,64,32] -> Cat[32+64=96] -> Project 96->32->1

#### Training Data: CAMELS LH IllustrisTNG z=0.00
- Input: Maps_Mcdm (dark matter only), 15000 x 256 x 256
- Target: Maps_Mtot (total matter = DM + baryons), same shape
- Parameters: params_LH_IllustrisTNG.txt, 1000 rows x 6 columns
  (Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2)
- Each simulation has 15 axis projections; params are repeated 15x
  to match (this was a critical bug fix -- previously fell through
  to random synthetic params, making conditioning useless)

#### Loss Function: BaryonicEmulatorLoss
Combined weighted loss:
- Pixel MSE (1.0): field-level accuracy
- Power spectrum (0.5): log P(k) matching in 32 k-bins
- Field statistics (0.1): mean, variance, skewness, kurtosis
- Gradient (0.05): Sobel-filtered edge preservation
- Multi-scale (0.1): avg-pool at scales 1,2,4,8
- Mass conservation (0.01): enforces Mtot ~ Mcdm * (1 + f_baryon)

#### Training Configuration
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
- Scheduler: 5-epoch linear warmup -> cosine annealing to 1e-6
- Batch size: 16
- Gradient clipping: 1.0
- Augmentation: random rot90, horizontal/vertical flips
- AMP disabled (complex FFT ops in SpectralConv2d)
- Early stopping: patience 15
- Checkpointing every epoch (full state for resume)

#### Data Normalization
Both input and target maps go through log1p transform before the network.
Output is transformed back via expm1. This stabilizes training on the
high dynamic range (~5 orders of magnitude) of density fields.

### 3. Streamlit Dashboard (webapp.py)

Tabs:
- CAMELS Data: browse maps, run inference, compare to ground truth
- N-Body Simulator: generate custom DM maps via bayronik-core binary
- Parameter Sweep: vary one param and see output field changes
- About: architecture description

Analysis metrics:
- Power spectrum P(k): isotropic 2D, computed on log1p field
- Baryon suppression S(k) = P_tot(k) / P_DM(k)
- 1-point PDF of log-density
- Log MSE, Log MAE, pixel relative error

Loads model directly (no server needed). Interactive Plotly heatmaps
with dark theme, custom diverging colormap for difference maps.

### 4. FastAPI Server (server.py)

REST API for inference. Required by the Rust/WASM frontend.
Endpoints: /infer (JSON), /infer_npy (binary), /health.
Uses lifespan context manager for model loading.

### 5. Rust TUI (bayronik-infer)

Terminal-based viewer using tch-rs (libtorch) and ratatui.
Braille-character heatmaps in the terminal. Supports both CAMELS
maps and on-the-fly N-body generation. Uses TorchScript model.

### 6. Rust/WASM Frontend (bayronik-web)

egui-based desktop and WASM app. Calls server.py for inference.
Currently a skeleton: has parameter sliders, heatmap display,
and synthetic test input, but no data loading, no analysis plots,
no N-body integration. ~40% feature parity with Streamlit.

## Training History

### Run 1 (Feb 2026, GCP L4)
- Dataset: CAMELS LH IllustrisTNG
- Bug: params_LH_IllustrisTNG.txt has 1000 rows, maps have 15000
  samples. Code checked `params.shape[0] == num_samples` (1000 != 15000),
  then `params.shape[0] > num_samples` (1000 > 15000, false), and fell
  through to generating random synthetic parameters. The model was
  trained on garbage conditioning inputs.
- Result: ~22% pixel relative error. Parameters had no effect on output.
- Loss weights: spectral=0.1, mass=0.0

### Run 2 (planned)
- Fix: np.repeat(params, 15, axis=0) to expand 1000 rows to 15000
- Improved loss: spectral=0.5, mass=0.01
- Batch size: 16 (up from 8, fits on L4 24GB)
- Expected: significantly better parameter sensitivity and lower error

## What's On Disk

```
bayronik/
  bayronik-core/        Rust PM N-body (physics bugs fixed, see above)
  bayronik-model/
    src/bayronik_model/  Python package (ufno, fno, losses, dataset, model, export)
    train.py             Training script
    webapp.py            Streamlit dashboard
    server.py            FastAPI inference server
    download_data.py     CAMELS data downloader
    data/                Maps + params (gitignored)
    weights/             Trained model weights (gitignored)
  bayronik-infer/        Rust TUI with tch-rs
  bayronik-web/          Rust/egui desktop + WASM frontend
  Makefile               Build, train, demo, GCP orchestration
  GUIDE.md               User guide for dashboard
  docs/EXPERIMENT.md     This file
```
