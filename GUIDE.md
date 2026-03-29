# Bayronik User Guide

## Running the App

### Desktop (recommended for local use)

```bash
# Start the inference server
make server

# In another terminal, run the desktop app
cd bayronik-web && cargo run --release
```

Or use `make demo` to start both together.

### Browser (WASM)

```bash
# Build WASM and serve
make serve-web
```

Open http://localhost:8080 in your browser. Requires the inference server running on port 8000.

## Tabs

### N-Body Simulator

Generate custom dark matter maps using the Rust Particle-Mesh N-body simulator. The simulation runs entirely client-side (in WASM for browser, native for desktop) — no server needed.

1. Set grid resolution (32³ or 64³), box size, time steps, and seed
2. Click "Run N-Body + Emulator"
3. The simulator generates a DM density map, sends it to the server for emulation

Displays:
- N-Body DM density map
- Emulated total matter map
- Baryonic effect (difference) with diverging colormap
- Power spectrum P(k), baryon suppression S(k), 1-point PDF

### CAMELS Data

Browse real CAMELS simulation maps and run inference:

1. Select a sample index (0 to 14999 for LH data)
2. Click "Load Sample" to fetch from the server
3. Click "Run Inference"

Displays:
- Input (Mcdm): Dark matter density from gravity-only simulation
- Predicted (Mtot): Total matter from the emulator
- Baryonic Effect: Difference map (diverging colormap, zero-centered)
- Ground Truth comparison with error map
- Metrics: Log MSE, Log MAE
- Analysis plots: P(k), S(k), PDF

### Parameter Sweep

Explore how the emulator responds to different feedback parameters:

1. Load a sample (from CAMELS or N-Body) first
2. Select a parameter to vary and number of steps
3. Click "Run Sweep"

Shows side-by-side output maps and overlaid power spectra. Try sweeping Omega_m or sigma_8 for the most dramatic visual differences.

### About

Project information and physics parameter definitions.

## Controls

- **Log scale**: Toggle log-space display (recommended for density fields)
- **Colormap**: Visual color scheme (Inferno default, also Viridis, Plasma, Magma, DarkDiverging)
- **Physics Parameters**: Six values for cosmological and feedback conditioning:
  - Omega_m: Total matter density parameter
  - sigma_8: Amplitude of matter fluctuations
  - A_SN1, A_SN2: Supernova feedback strength
  - A_AGN1, A_AGN2: AGN feedback strength

## Interpreting Results

### Baryonic Effect Map

The difference map (Mtot - Mcdm) uses a dark diverging colormap centered at zero:

- **Red regions**: Baryons increase density (gas cooling into halo centers)
- **Blue regions**: Baryons decrease density (AGN feedback ejecting gas)
- **Dark/black**: No significant baryonic effect

### Power Spectrum P(k)

The 2D isotropic power spectrum shows how much structure exists at each scale:
- Low k (left): Large scales (cosmic web, voids)
- High k (right): Small scales (halo cores, substructure)
- Prediction should match ground truth across all k

### Baryon Suppression S(k)

S(k) = P_total(k) / P_DM(k) quantifies the baryonic correction as a function of scale:
- S(k) = 1: No baryonic effect
- S(k) > 1: Baryons enhance power (gas cooling)
- S(k) < 1: Baryons suppress power (feedback)
- Typically S(k) dips below 1 at k ~ 1-10 (feedback-dominated scales)

### 1-Point PDF

The pixel value distribution in log-density space. A good emulator reproduces the full PDF shape, including the high-density tail (halos) and low-density void regions.

## Training

See the main [README.md](README.md) for training instructions, or run:

```bash
make train
```
