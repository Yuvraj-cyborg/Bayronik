# Bayronik: Baryonic Field Emulator for Weak Lensing

A high-performance, multi-fidelity field-level emulator that generates realistic 2D convergence (κ) maps including baryonic effects, without running expensive hydrodynamic simulations.

## Scientific Goal

Baryonic feedback (AGN, supernovae) redistributes matter on ~kpc–Mpc scales and biases weak-lensing cosmology. **Bayronik** emulates these effects instantly using machine learning, enabling:
- Fast forward modeling for likelihood-free inference
- Survey forecasts and instrument studies  
- Field-level analysis (peaks, Minkowski functionals, higher-order moments)

### Why Field-Level?
Summary statistics (power spectra) discard information. Field-level emulators produce maps compatible with many statistics and can be forward-modeled into realistic observations.

### Documentation
- **[GUIDE.md](GUIDE.md)** - Complete TUI visualization guide (how to read colors, statistics, physics)
- **[README.md](README.md)** - This file (setup, architecture, data access)

## Project Structure

```
bayronik/
├── bayronik-core/      # Rust N-body simulation (gravity-only, PM method)
├── bayronik-infer/     # Rust TUI/CLI for inference with trained models
├── bayronik-model/     # Python ML training pipeline (PyTorch U-Net)
└── README.md
```

##  Quick Start

### Prerequisites

- **Rust** (1.70+): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Python** (3.9+) with PyTorch
- **LibTorch** (for Rust inference): Provided via PyTorch

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bayronik.git
cd bayronik

# Set up Python environment
cd bayronik-model
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download CAMELS data (see Data Access section)
cd data/
# Place your .npy or .hdf5 files here
```

### Training the Emulator

```bash
cd bayronik-model
source .venv/bin/activate
python train.py

# Export to TorchScript for Rust inference
python exporter.py
```

### Running the TUI Inference Tool

```bash
cd ../bayronik-infer

# Activate Python env (needed for LibTorch)
source ../bayronik-model/.venv/bin/activate

# Set environment variables
export DYLD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LIBTORCH_USE_PYTORCH=1

# Build and run
cargo build --release
cargo run --release
```

**Or use the launch script**:
```bash
cd ~/Projects/bayronik
./run_bayronik.sh
```

**Interactive controls**:
- `→` or `n` - Next simulation
- `←` or `p` - Previous simulation  
- `r` - Random simulation
- `q` - Quit

** [Complete TUI Guide →](GUIDE.md)** - Learn how to interpret the visualizations!
```

**Pro tip**: Create an alias in your shell:
```bash
alias bayronik-infer='cd bayronik/bayronik-infer && source ../bayronik-model/.venv/bin/activate && export DYLD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))") && export LIBTORCH_USE_PYTORCH=1 && cargo run --release'
```

## Data Access

### CAMELS Multifield Dataset (CMD)

Official documentation: https://camels.readthedocs.io/

**Current setup** (NPY files):
- Download from: https://camels-multifield-dataset.readthedocs.io/
- Place in `bayronik-model/data/`:
  - `Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy` (Dark matter)
  - `Maps_Mtot_IllustrisTNG_CV_z=0.00.npy` (Total matter)

**Recommended** (HDF5 format - smaller, faster):
```bash
# Request access at: https://camels.readthedocs.io/en/latest/data_access.html
cd bayronik-model/data
wget --user=<username> --password=<password> \
  https://users.flatironinstitute.org/~fvillaescusa/priv/CMD_IllustrisTNG/Maps_*_CV_z=0.00.hdf5
```

## Architecture Details

### bayronik-model (Python)
- **Model**: U-Net (encoder-decoder with skip connections)
- **Input**: Dark matter 2D projected maps (256×256)
- **Output**: Total matter maps (including baryons)
- **Training**: MSE loss, Adam optimizer, W&B logging
- **Export**: TorchScript (.pt) for Rust interop

### bayronik-infer (Rust)
- **Purpose**: Fast inference and visualization
- **Features**:
  - Load TorchScript models via `tch-rs`
  - ASCII heatmap visualization in terminal
  - Planned: Parameter controls, power spectrum plots
- **Performance**: <1s per map on CPU

### bayronik-core (Rust)
- **Purpose**: Generate cheap gravity-only fields for multi-fidelity training
- **Method**: Particle-Mesh N-body simulation
- **Features**:
  - FFT-based Poisson solver (3D)
  - Periodic boundary conditions
  - Planned: Cosmological integration, ICs from P(k)

## 🔬 Current Status (v0.1.0-alpha)

### ✅ Working
- [x] Basic U-Net training on CAMELS data
- [x] TorchScript export
- [x] Rust inference with TUI
- [x] ASCII visualization

### 🚧 In Progress
- [ ] Validation loop and metrics
- [ ] Parameter conditioning (Ωₘ, σ₈, A_AGN, etc.)
- [ ] HDF5 data loader
- [ ] Power spectrum computation
- [ ] Better visualization (color gradients)

### 📋 Planned
- [ ] Multi-fidelity training (gravity-only + hydro residual)
- [ ] Uncertainty quantification
- [ ] Cosmological N-body improvements
- [ ] Python bindings (PyO3)
- [ ] WASM demo website
- [ ] Docker container

## 🤝 Contributing

This is an active research project. Contributions welcome!

Areas for improvement:
1. Multi-field inputs (gas, stars, not just dark matter)
2. Conditional generation (vary cosmo/baryon params)
3. Validation metrics (power spectrum comparison, peak counts)
4. Better visualizations
5. Documentation and examples

## 📖 References

Key papers that inspired this work:

- [CAMELS Dataset](https://arxiv.org/abs/2109.10915): Villaescusa-Navarro et al. (2022)
- [Field-Level Emulation](https://arxiv.org/abs/2207.05202): Jamieson et al. (2022)
- [Baryonic Corrections](https://academic.oup.com/mnras/article/526/3/3396/7280363): Sharma et al. (2023)
- [Multi-Fidelity ML](https://arxiv.org/abs/2105.01081): Ho et al. (2021)

## 📄 License

MIT License

## Acknowledgments

- **CAMELS Collaboration** for simulation data
- **Flatiron Institute** for hosting the dataset
- **tch-rs** for PyTorch-Rust bindings
- **ratatui** for terminal UI framework

---

**Status**: Research prototype (v0.1.0-alpha)  
**Contact**: yuvrajbiswalofficial@gmail.com
**Last updated**: October 2025

