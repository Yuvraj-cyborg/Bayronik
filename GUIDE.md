# Bayronik User Guide

## Streamlit Dashboard

The primary interface is a Streamlit web app. Launch it with:

```bash
make demo
```

Open http://localhost:8501 in your browser.

### CAMELS Data Tab

Browse CAMELS simulation maps and run inference:

1. Select a sample index (0 to 14999 for LH data)
2. Choose "Use sample's actual parameters" to use the simulation's true params, or uncheck to use the manual sliders
3. Click "Run Inference"

The dashboard shows:
- **Input (Mcdm)**: Dark matter density from gravity-only simulation
- **Predicted (Mtot)**: Total matter (DM + baryons) from the emulator
- **Baryonic Effect**: Difference map showing where baryons redistribute matter

If ground truth data (Maps_Mtot) is available, additional panels show:
- Ground truth vs prediction comparison
- Error metrics (Log MSE, Log MAE, relative error)
- Power spectrum P(k) comparison
- Baryon suppression ratio S(k) = P_tot/P_DM
- 1-point pixel PDF

### N-Body Simulator Tab

Generate custom dark matter maps using the Rust N-body simulator:

1. Set grid resolution, box size, and time steps
2. Click "Run N-Body + Emulator"
3. The simulator generates a DM density map, which is fed through the emulator

Requires building the binary first: `make build-nbody`

### Parameter Sweep Tab

Explore how the emulator responds to different feedback parameters:

1. Select a sample and parameter to vary
2. Choose number of steps
3. Click "Run Sweep"

Shows side-by-side maps and overlaid power spectra.

### Sidebar Controls

- **Log scale**: Toggle log-space display (recommended for density fields)
- **Colormap**: Visual color scheme
- **Physics Parameters**: Six sliders for cosmological and feedback parameters:
  - Omega_m: Total matter density
  - sigma_8: Amplitude of fluctuations
  - A_SN1, A_SN2: Supernova feedback strength
  - A_AGN1, A_AGN2: AGN feedback strength

## Rust TUI

For terminal-based visualization:

```bash
make infer
```

Controls:
- Right/Left arrows or n/p: Navigate simulations
- r: Random sample
- g: Generate on-the-fly N-body map
- c: Switch to CAMELS maps
- q: Quit

Requires libtorch (provided by PyTorch installation).

## Interpreting Results

### Baryonic Effect Map

The difference map (Mtot - Mcdm) reveals where baryonic physics modifies the matter distribution:

- **Bright regions**: Baryons increase density (gas cooling into halo centers)
- **Dark regions**: Baryons decrease density (AGN feedback ejecting gas)
- **Neutral**: No significant baryonic effect

### Power Spectrum P(k)

The 2D isotropic power spectrum shows how much structure exists at each scale:
- Low k: Large scales (cosmic web, voids)
- High k: Small scales (halo cores, substructure)
- Prediction should match ground truth across all k

### Baryon Suppression S(k)

S(k) = P_total(k) / P_DM(k) quantifies the baryonic correction as a function of scale:
- S(k) = 1: No baryonic effect
- S(k) > 1: Baryons enhance power (gas cooling)
- S(k) < 1: Baryons suppress power (feedback)
- Typically S(k) dips below 1 at k ~ 1-10 (feedback-dominated scales)

### 1-Point PDF

The pixel value distribution in log-density space. A good emulator should reproduce the full PDF shape, including the high-density tail (halos) and low-density void regions.

## Training

See [bayronik-model/README.md](bayronik-model/README.md) for training instructions.

## References

- CAMELS: https://camels.readthedocs.io
- CAMELS Multifield Dataset: arXiv:2109.10915
- U-FNO: doi:10.1016/j.advwatres.2022.104180
- Baryonic effects review: arXiv:1510.06034
