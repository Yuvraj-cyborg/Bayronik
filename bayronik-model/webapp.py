#!/usr/bin/env python3
"""Streamlit dashboard for the Bayronik baryonic field emulator."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model.ufno import UFNO2dConditional

COLORMAPS = ["Inferno", "Viridis", "Plasma", "Magma", "Hot", "Cividis"]

DARK_DIVERGING = [
    [0.0, "rgb(20,60,180)"],
    [0.3, "rgb(40,40,100)"],
    [0.5, "rgb(15,15,15)"],
    [0.7, "rgb(100,35,35)"],
    [1.0, "rgb(200,30,30)"],
]

PROJECT_ROOT = Path(__file__).parent.parent
NBODY_BIN = PROJECT_ROOT / "target" / "release" / "examples" / "generate_map"


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).parent / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"

    if not model_path.exists():
        st.error(f"Model weights not found at `{model_path}`")
        return None, device

    model = UFNO2dConditional(
        in_channels=1, out_channels=1,
        base_channels=32, modes=32, depth=4, num_conditions=6,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_test_data():
    data_dir = Path(__file__).parent / "data"
    available = {}

    for dt in ["LH", "CV"]:
        dm = data_dir / f"Maps_Mcdm_IllustrisTNG_{dt}_z=0.00.npy"
        mt = data_dir / f"Maps_Mtot_IllustrisTNG_{dt}_z=0.00.npy"
        if dm.exists() and "Mcdm" not in available:
            print(f"[load] Memory-mapping {dm.name} ...")
            available["Mcdm"] = np.load(dm, mmap_mode="r")
            available["dataset_type"] = dt
        if mt.exists() and "Mtot" not in available:
            print(f"[load] Memory-mapping {mt.name} ...")
            available["Mtot"] = np.load(mt, mmap_mode="r")

    for name in ["params_LH_IllustrisTNG.txt", "params_IllustrisTNG_LH.txt"]:
        p = data_dir / name
        if p.exists():
            available["params"] = np.loadtxt(p)
            break

    print(f"[load] Data ready: {list(available.keys())}")
    return available


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(model, device, input_map: np.ndarray, params: dict) -> np.ndarray:
    """Run U-FNO inference on a single 2D density map."""
    safe_input = np.clip(input_map.astype(np.float32), 0, None)
    input_log = np.log1p(safe_input)
    input_tensor = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(device)
    conditions = torch.tensor([list(params.values())], dtype=torch.float32).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor, conditions)

    return np.expm1(output_tensor.squeeze().cpu().numpy())


def run_nbody(grid_res: int = 64, box_size: float = 100.0,
              steps: int = 10, proj_res: int = 256, seed: int = 42):
    """Run N-body simulation via the bayronik-core binary."""
    if not NBODY_BIN.exists():
        return None, (
            "N-body binary not found. Build it first:\n\n"
            "```\nmake build-nbody\n```"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = f"nbody_{proj_res}.npy"
        try:
            result = subprocess.run(
                [
                    str(NBODY_BIN),
                    str(grid_res), str(box_size),
                    str(steps), str(proj_res), out_file, str(seed),
                ],
                capture_output=True, text=True, timeout=120,
                cwd=tmpdir,
            )
            npy_path = Path(tmpdir) / out_file
            if npy_path.exists():
                data = np.load(npy_path)
                return data.reshape(proj_res, proj_res), None
            return None, f"Simulation exited but no output. stderr: {result.stderr}"
        except subprocess.TimeoutExpired:
            return None, "Simulation timed out (>120s)"
        except Exception as e:
            return None, str(e)


# ---------------------------------------------------------------------------
# Cosmology analysis functions
# ---------------------------------------------------------------------------

def _safe_log1p(field: np.ndarray) -> np.ndarray:
    """log1p that clamps non-positive values to avoid -inf / nan."""
    f = np.asarray(field, dtype=np.float64)
    f = np.clip(f, 0, None)
    return np.log1p(f)


def compute_power_spectrum(field: np.ndarray):
    """Isotropic 2D power spectrum P(k)."""
    f = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    n = f.shape[0]
    fft = np.fft.fft2(f)
    pk2d = np.abs(fft) ** 2 / n**4

    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    kx2d, ky2d = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx2d**2 + ky2d**2)

    k_bins = np.arange(1, n // 2)
    pk = np.zeros(len(k_bins) - 1)
    k_centers = np.zeros(len(k_bins) - 1)

    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        if mask.sum() > 0:
            pk[i] = pk2d[mask].mean()
            k_centers[i] = k_mag[mask].mean()

    valid = pk > 0
    return k_centers[valid], pk[valid]


def compute_baryon_suppression(k_dm, pk_dm, k_tot, pk_tot):
    """S(k) = P_tot(k) / P_dm(k), interpolated to common k grid."""
    if len(k_dm) == 0 or len(k_tot) == 0:
        return np.array([1.0]), np.array([1.0])

    k_min = max(k_dm[0], k_tot[0])
    k_max = min(k_dm[-1], k_tot[-1])
    if k_max <= k_min:
        return np.array([1.0]), np.array([1.0])

    k_common = np.linspace(k_min, k_max, min(len(k_dm), len(k_tot)))

    pk_dm_interp = np.interp(k_common, k_dm, pk_dm)
    pk_tot_interp = np.interp(k_common, k_tot, pk_tot)

    with np.errstate(divide="ignore", invalid="ignore"):
        suppression = np.where(pk_dm_interp > 0,
                               pk_tot_interp / pk_dm_interp, 1.0)
    return k_common, suppression


def compute_pixel_pdf(field: np.ndarray, n_bins: int = 80):
    """Compute 1-point PDF of the log-density field."""
    log_field = _safe_log1p(field.flatten())
    finite = log_field[np.isfinite(log_field)]
    if len(finite) == 0:
        return np.array([0.0]), np.array([0.0])
    counts, edges = np.histogram(finite, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_heatmap(data, title, cmap="Inferno", log_scale=True, diverging=False):
    bg = "#0e1117"
    tc = "#cccccc"

    if diverging:
        vmax = float(np.percentile(np.abs(data), 99))
        plot_data = data.copy()
        colorscale = DARK_DIVERGING
        zmin, zmax = -vmax, vmax
        cbar_title = "delta"
    else:
        if log_scale and data.min() >= 0:
            plot_data = np.log10(data + 1)
        else:
            plot_data = data
        colorscale = cmap
        zmin, zmax = None, None
        cbar_title = "log10" if log_scale else "value"

    fig = go.Figure(data=go.Heatmap(
        z=plot_data,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(
            thickness=10, len=0.85,
            tickfont=dict(color=tc, size=9),
            title=dict(text=cbar_title, font=dict(color=tc, size=9)),
        ),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=tc, size=13), x=0.5),
        paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(visible=False, scaleanchor="x", constrain="domain"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=320,
    )
    return fig


def make_line_plot(traces, title, xlabel, ylabel,
                   logx=True, logy=True, hline=None):
    """Generic multi-trace line plot for dark theme."""
    bg = "#0e1117"
    tc = "#cccccc"
    palette = ["#ff6b35", "#4ecdc4", "#f7fff7", "#ffe66d", "#ff4444",
               "#a855f7", "#22d3ee", "#f472b6"]

    fig = go.Figure()
    for i, (x, y, name) in enumerate(traces):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash",
                      line_color="#666666", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color=tc, size=14), x=0.5),
        xaxis=dict(
            title=xlabel, color=tc,
            type="log" if logx else "linear",
            gridcolor="#262730", showgrid=True,
        ),
        yaxis=dict(
            title=ylabel, color=tc,
            type="log" if logy else "linear",
            gridcolor="#262730", showgrid=True,
        ),
        paper_bgcolor=bg, plot_bgcolor=bg,
        legend=dict(font=dict(color=tc, size=11)),
        margin=dict(l=60, r=20, t=40, b=50),
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(test_data):
    dataset_type = test_data.get("dataset_type", "Unknown")
    has_lh_params = "params" in test_data

    with st.sidebar:
        st.header("Display")
        log_scale = st.checkbox("Log scale", value=True)
        cmap = st.selectbox("Colormap", COLORMAPS, index=0)

        st.divider()
        st.header("Physics Parameters")

        if dataset_type == "LH" and has_lh_params:
            use_sample = st.checkbox("Use sample's actual parameters", value=True)
        else:
            use_sample = False

        omega_m = st.slider("Omega_m", 0.1, 0.5, 0.3, 0.01)
        sigma_8 = st.slider("sigma_8", 0.6, 1.0, 0.8, 0.01)
        st.divider()
        a_sn1 = st.slider("A_SN1 (stellar)", 0.25, 4.0, 1.0, 0.05)
        a_agn1 = st.slider("A_AGN1 (AGN)", 0.25, 4.0, 1.0, 0.05)
        a_sn2 = st.slider("A_SN2", 0.5, 2.0, 1.0, 0.05)
        a_agn2 = st.slider("A_AGN2", 0.5, 2.0, 1.0, 0.05)

    manual_params = {
        "omega_m": omega_m, "sigma_8": sigma_8,
        "a_sn1": a_sn1, "a_agn1": a_agn1,
        "a_sn2": a_sn2, "a_agn2": a_agn2,
    }
    return log_scale, cmap, use_sample, manual_params


def get_sample_params(test_data, sample_idx, n_samples):
    """Look up the actual simulation parameters for a given sample index."""
    params_arr = test_data["params"]
    n_sims = len(params_arr)
    maps_per_sim = max(1, n_samples // n_sims) if n_sims > 0 else 1
    sim_idx = min(sample_idx // maps_per_sim, n_sims - 1)
    sp = params_arr[sim_idx]
    return {
        "omega_m": float(sp[0]), "sigma_8": float(sp[1]),
        "a_sn1": float(sp[2]) if len(sp) > 2 else 1.0,
        "a_agn1": float(sp[3]) if len(sp) > 3 else 1.0,
        "a_sn2": float(sp[4]) if len(sp) > 4 else 1.0,
        "a_agn2": float(sp[5]) if len(sp) > 5 else 1.0,
    }, sim_idx


# ---------------------------------------------------------------------------
# Tab: CAMELS Data
# ---------------------------------------------------------------------------

def tab_camels(model, device, test_data, log_scale, cmap,
               use_sample, manual_params):
    if "Mcdm" not in test_data:
        st.error("No test data found. Run `make download-lh`")
        return

    dm_maps = test_data["Mcdm"]
    n_samples = dm_maps.shape[0] if dm_maps.ndim == 3 else 1

    sample_idx = st.number_input("Sample index", 0, n_samples - 1, 0)
    input_map = np.array(dm_maps[sample_idx]) if dm_maps.ndim == 3 else np.array(dm_maps)

    gt_map = None
    if "Mtot" in test_data:
        m = test_data["Mtot"]
        gt_map = np.array(m[sample_idx]) if m.ndim == 3 else np.array(m)

    if use_sample and "params" in test_data:
        params, sim_idx = get_sample_params(test_data, sample_idx, n_samples)
        st.info(
            f"Sim {sim_idx}: Om={params['omega_m']:.3f}  "
            f"s8={params['sigma_8']:.3f}  "
            f"ASN1={params['a_sn1']:.2f}  AAGN1={params['a_agn1']:.2f}"
        )
    else:
        params = manual_params

    if st.button("Run Inference", type="primary", key="data_infer"):
        if model is None:
            st.error("Model not loaded")
            return
        with st.spinner("Running inference..."):
            try:
                output_map = run_inference(model, device, input_map, params)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                import traceback; traceback.print_exc()
                return
        st.session_state["d_input"] = input_map
        st.session_state["d_output"] = output_map
        st.session_state["d_diff"] = output_map - input_map
        st.session_state["d_gt"] = gt_map

    if "d_input" not in st.session_state:
        st.info("Select a sample above and click **Run Inference** to begin.")
        return

    inp = st.session_state["d_input"]
    out = st.session_state["d_output"]
    diff = st.session_state["d_diff"]

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(make_heatmap(inp, "Input: Mcdm", cmap, log_scale),
                    width="stretch")
    c2.plotly_chart(make_heatmap(out, "Predicted: Mtot", cmap, log_scale),
                    width="stretch")
    c3.plotly_chart(make_heatmap(diff, "Baryonic Effect", cmap, diverging=True),
                    width="stretch")

    if st.session_state["d_gt"] is None:
        return

    gt = st.session_state["d_gt"]
    error = out - gt

    st.subheader("Ground Truth Comparison")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(make_heatmap(gt, "Ground Truth: Mtot", cmap, log_scale),
                    width="stretch")
    c2.plotly_chart(make_heatmap(out, "Prediction", cmap, log_scale),
                    width="stretch")
    c3.plotly_chart(make_heatmap(error, "Error", cmap, diverging=True),
                    width="stretch")

    gt_log = np.log10(gt + 1)
    out_log = np.log10(out + 1)
    log_err = out_log - gt_log

    c1, c2, c3 = st.columns(3)
    c1.metric("Log MSE", f"{np.mean(log_err**2):.4f}")
    c2.metric("Log MAE", f"{np.mean(np.abs(log_err)):.4f}")
    c3.metric("Pixel Rel Error",
              f"{np.mean(np.abs(error) / (np.abs(gt) + 1e-8)):.2%}")

    render_analysis(inp, out, gt, log_scale)


def render_analysis(inp, out, gt, log_scale):
    """Power spectrum, baryon suppression, and pixel PDF comparison."""
    st.subheader("Power Spectrum P(k)")
    k_inp, pk_inp = compute_power_spectrum(_safe_log1p(inp))
    k_out, pk_out = compute_power_spectrum(_safe_log1p(out))
    k_gt, pk_gt = compute_power_spectrum(_safe_log1p(gt))

    fig_ps = make_line_plot(
        [(k_inp, pk_inp, "Input (Mcdm)"),
         (k_out, pk_out, "Prediction"),
         (k_gt, pk_gt, "Ground Truth (Mtot)")],
        "Power Spectrum P(k)", "k", "P(k)",
    )
    st.plotly_chart(fig_ps, width="stretch")

    st.subheader("Baryon Suppression Ratio S(k)")
    st.caption("S(k) = P_total(k) / P_DM(k).  S=1 means no baryonic effect.")
    c1, c2 = st.columns(2)
    with c1:
        k_s_pred, s_pred = compute_baryon_suppression(k_inp, pk_inp, k_out, pk_out)
        k_s_gt, s_gt = compute_baryon_suppression(k_inp, pk_inp, k_gt, pk_gt)
        fig_supp = make_line_plot(
            [(k_s_pred, s_pred, "Predicted S(k)"),
             (k_s_gt, s_gt, "True S(k)")],
            "Baryon Suppression", "k", "S(k) = P_tot/P_DM",
            logy=False, hline=1.0,
        )
        st.plotly_chart(fig_supp, width="stretch")

    with c2:
        x_inp, y_inp = compute_pixel_pdf(inp)
        x_out, y_out = compute_pixel_pdf(out)
        x_gt, y_gt = compute_pixel_pdf(gt)
        fig_pdf = make_line_plot(
            [(x_inp, y_inp, "Input (Mcdm)"),
             (x_out, y_out, "Prediction"),
             (x_gt, y_gt, "Ground Truth")],
            "1-Point PDF", "log(1+rho)", "density",
            logx=False, logy=False,
        )
        st.plotly_chart(fig_pdf, width="stretch")


# ---------------------------------------------------------------------------
# Tab: N-Body Simulator
# ---------------------------------------------------------------------------

def tab_nbody(model, device, log_scale, cmap, manual_params):
    st.subheader("N-Body Dark Matter Simulation")
    st.markdown(
        "Generate a custom dark matter density map using the Particle-Mesh "
        "N-body simulator (`bayronik-core`), then run the emulator on it."
    )

    if not NBODY_BIN.exists():
        st.warning("N-body binary not found. Run `make build-nbody` first.")

    c1, c2, c3, c4 = st.columns(4)
    grid_res = c1.select_slider("Grid resolution", [32, 64, 128], value=64)
    box_size = c2.slider("Box size (Mpc/h)", 50.0, 500.0, 100.0, 25.0)
    n_steps = c3.slider("Time steps", 5, 50, 10)
    seed = c4.number_input("RNG Seed", value=42, min_value=0, max_value=99999)

    if st.button("Run N-Body + Emulator", type="primary", key="nbody_run"):
        with st.spinner("Running N-body simulation..."):
            nbody_map, err = run_nbody(grid_res, box_size, n_steps, seed=int(seed))
        if err:
            st.error(err)
            return
        with st.spinner("Running emulator..."):
            output_map = run_inference(model, device, nbody_map, manual_params)

        st.session_state["nb_input"] = nbody_map
        st.session_state["nb_output"] = output_map
        st.session_state["nb_diff"] = output_map - nbody_map

    if "nb_input" not in st.session_state:
        return

    inp = st.session_state["nb_input"]
    out = st.session_state["nb_output"]
    diff = st.session_state["nb_diff"]

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(make_heatmap(inp, "N-Body: DM Density", cmap, log_scale),
                    width="stretch")
    c2.plotly_chart(make_heatmap(out, "Emulated: Total Matter", cmap, log_scale),
                    width="stretch")
    c3.plotly_chart(make_heatmap(diff, "Baryonic Effect", cmap, diverging=True),
                    width="stretch")

    st.subheader("Power Spectrum")
    k_inp, pk_inp = compute_power_spectrum(_safe_log1p(inp))
    k_out, pk_out = compute_power_spectrum(_safe_log1p(out))

    c1, c2 = st.columns(2)
    with c1:
        fig_ps = make_line_plot(
            [(k_inp, pk_inp, "N-Body DM"), (k_out, pk_out, "Emulated Mtot")],
            "Power Spectrum", "k", "P(k)",
        )
        st.plotly_chart(fig_ps, width="stretch")
    with c2:
        k_s, s_k = compute_baryon_suppression(k_inp, pk_inp, k_out, pk_out)
        fig_sup = make_line_plot(
            [(k_s, s_k, "S(k)")],
            "Baryon Suppression", "k", "S(k)",
            logy=False, hline=1.0,
        )
        st.plotly_chart(fig_sup, width="stretch")


# ---------------------------------------------------------------------------
# Tab: Parameter Sweep
# ---------------------------------------------------------------------------

def tab_sweep(model, device, test_data, log_scale, cmap, manual_params):
    st.subheader("Parameter Sensitivity")
    st.markdown("Vary one parameter and observe how the output field changes.")

    if "Mcdm" not in test_data:
        st.warning("Load CAMELS data first: `make download-lh`")
        return

    dm_maps = test_data["Mcdm"]
    sweep_idx = st.number_input("Sample", 0, dm_maps.shape[0] - 1, 0,
                                key="sweep_idx")
    sweep_input = np.array(dm_maps[sweep_idx])

    c1, c2 = st.columns(2)
    sweep_param = c1.selectbox(
        "Parameter to vary",
        ["a_sn1", "a_agn1", "omega_m", "sigma_8"])
    n_steps = c2.slider("Steps", 2, 5, 3)

    if not st.button("Run Sweep", type="primary", key="sweep_run"):
        return

    ranges = {
        "omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0),
        "a_sn1": (0.25, 4.0), "a_agn1": (0.25, 4.0),
        "a_sn2": (0.5, 2.0), "a_agn2": (0.5, 2.0),
    }
    values = np.linspace(*ranges[sweep_param], n_steps)
    cols = st.columns(n_steps)

    spectra = []
    for i, val in enumerate(values):
        tp = manual_params.copy()
        tp[sweep_param] = float(val)
        result = run_inference(model, device, sweep_input, tp)
        with cols[i]:
            st.plotly_chart(
                make_heatmap(result, f"{val:.2f}", cmap, log_scale),
                width="stretch",
            )
            st.caption(f"mean={result.mean():.2e}")

        k, pk = compute_power_spectrum(_safe_log1p(result))
        spectra.append((k, pk, f"{sweep_param}={val:.2f}"))

    st.subheader("Power Spectrum vs Parameter")
    fig_ps = make_line_plot(spectra, "Power Spectrum", "k", "P(k)")
    st.plotly_chart(fig_ps, width="stretch")


# ---------------------------------------------------------------------------
# Tab: About
# ---------------------------------------------------------------------------

def tab_about():
    st.subheader("About Bayronik")
    st.markdown("""
**Bayronik** is an interactive field-level baryonic emulator for weak lensing cosmology.

**The problem**: Weak lensing surveys (Euclid, LSST, Roman) need to account for
baryonic feedback (AGN, supernovae) that redistributes matter on kpc-Mpc scales.
Full hydrodynamic simulations take months on supercomputers. Bayronik does it in
milliseconds.

**Architecture**: U-FNO (U-Net enhanced Fourier Neural Operator) with FiLM
conditioning on 6 parameters: Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2.

**Training data**: CAMELS Multifield Dataset, IllustrisTNG Latin Hypercube suite
(1000 simulations x 15 projections = 15,000 256x256 maps).

**Pipeline**:

```
N-body PM simulation (bayronik-core, Rust)
        |
        v
  DM density map  +  [physics parameters]
        |
        v
  U-FNO emulator (PyTorch)
        |
        v
  Total matter map (DM + baryons)
        |
        v
  Validation: P(k), S(k), 1-point PDF
```

**Differentiation**:
| Project | Approach | Limitation |
|---------|----------|------------|
| BACCO | Power spectrum emulator | Not field-level |
| EMBER-2 | Field-level on FIRE-2 | Galaxy-scale, not cosmological |
| syren-baryon | Analytic fitting formulas | Not field-level |
| **Bayronik** | Field-level + interactive + N-body | Full pipeline |
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Bayronik", layout="wide")
    st.title("Bayronik")
    st.caption("Field-level baryonic emulator: dark matter density -> total matter")

    with st.spinner("Loading model..."):
        model, device = load_model()
    with st.spinner("Loading data (memory-mapped)..."):
        test_data = load_test_data()
    if model is None:
        st.error("Failed to load model. Check weights/ directory.")
        st.stop()
    st.sidebar.success(f"Model loaded on {device}")

    log_scale, cmap, use_sample, manual_params = render_sidebar(test_data)

    tabs = st.tabs(["CAMELS Data", "N-Body Simulator", "Parameter Sweep", "About"])

    with tabs[0]:
        tab_camels(model, device, test_data, log_scale, cmap,
                   use_sample, manual_params)
    with tabs[1]:
        tab_nbody(model, device, log_scale, cmap, manual_params)
    with tabs[2]:
        tab_sweep(model, device, test_data, log_scale, cmap, manual_params)
    with tabs[3]:
        tab_about()


if __name__ == "__main__":
    main()
