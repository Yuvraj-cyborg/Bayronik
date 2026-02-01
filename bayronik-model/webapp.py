#!/usr/bin/env python3
"""
Streamlit web app for Bayronik baryonic field emulator.

Run: uv run streamlit run webapp.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model.ufno import UFNO2dConditional

# Colormaps
COLORMAPS = {
    "inferno": "Inferno",
    "viridis": "Viridis", 
    "plasma": "Plasma",
    "magma": "Magma",
    "hot": "Hot",
    "cividis": "Cividis",
}


@st.cache_resource
def load_model():
    """Load and cache the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).parent / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        return None, device
    
    model = UFNO2dConditional(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        modes=32,
        depth=4,
        num_conditions=6,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, device


def load_test_data():
    """Load CAMELS test data if available."""
    data_dir = Path(__file__).parent / "data"
    
    available = {}
    
    for dataset_type in ["LH", "CV"]:
        dm_path = data_dir / f"Maps_Mcdm_IllustrisTNG_{dataset_type}_z=0.00.npy"
        mtot_path = data_dir / f"Maps_Mtot_IllustrisTNG_{dataset_type}_z=0.00.npy"
        
        if dm_path.exists() and "Mcdm" not in available:
            available["Mcdm"] = np.load(dm_path)
            available["dataset_type"] = dataset_type
        
        if mtot_path.exists() and "Mtot" not in available:
            available["Mtot"] = np.load(mtot_path)
    
    params_path = data_dir / "params_IllustrisTNG_LH.txt"
    if not params_path.exists():
        params_path = data_dir / "params_LH_IllustrisTNG.txt"
    
    if params_path.exists():
        available["params"] = np.loadtxt(params_path)
    
    return available


def run_inference(model, device, input_map, params):
    """Run model inference."""
    input_log = np.log1p(input_map.astype(np.float32))
    input_tensor = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(device)
    
    conditions = torch.tensor([list(params.values())], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor, conditions)
    
    output_log = output_tensor.squeeze().cpu().numpy()
    output_map = np.expm1(output_log)
    
    return output_map


def create_heatmap(data, title, colorscale="Inferno", log_scale=True, diverging=False, dark_theme=True):
    """Create interactive plotly heatmap."""
    
    if diverging:
        vmax = np.abs(data).max()
        plot_data = data
        colorscale = "RdBu_r" if dark_theme else "RdBu"
        zmin, zmax = -vmax, vmax
    else:
        if log_scale and data.min() >= 0:
            plot_data = np.log10(data + 1)
        else:
            plot_data = data
        zmin, zmax = None, None
    
    bg_color = "#0e1117" if dark_theme else "white"
    text_color = "white" if dark_theme else "black"
    
    fig = go.Figure(data=go.Heatmap(
        z=plot_data,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            tickfont=dict(color=text_color),
            title=dict(text="log10(value+1)" if log_scale and not diverging else "value", 
                      font=dict(color=text_color)),
        ),
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=text_color, size=16)),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Bayronik - Baryonic Field Emulator",
        layout="wide",
    )
    
    st.title("Bayronik - Baryonic Field Emulator")
    st.markdown("**U-FNO based emulator for mapping dark matter to total matter fields**")
    
    model, device = load_model()
    test_data = load_test_data()
    
    if model is None:
        st.stop()
    
    dataset_type = test_data.get("dataset_type", "Unknown")
    st.success(f"Model loaded on {device} | Test data: {dataset_type}")
    
    if dataset_type == "CV":
        st.warning("CV dataset has FIXED parameters. Download LH data for parameter sensitivity.")
        with st.expander("Download LH Data (click to expand)"):
            st.code("""
# Download from CAMELS directly using wget:
cd /Volumes/T7-SSD/Bayronik/bayronik-model/data

# Dark matter maps (LH = Latin Hypercube with varying params)
wget "https://users.flatironinstitute.org/~camels/Maps/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"

# Total matter maps  
wget "https://users.flatironinstitute.org/~camels/Maps/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"

# Parameter file
wget "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/params_IllustrisTNG_LH.txt" -O params_IllustrisTNG_LH.txt
            """, language="bash")
    
    has_lh_params = "params" in test_data
    
    with st.sidebar:
        st.header("Display Settings")
        dark_theme = st.checkbox("Dark theme", value=True)
        log_scale = st.checkbox("Log scale", value=True)
        colormap = st.selectbox("Colormap", list(COLORMAPS.keys()), index=0)
        
        st.markdown("---")
        st.header("Physics Parameters")
        
        if dataset_type == "LH" and has_lh_params:
            use_sample_params = st.checkbox("Use sample's actual parameters", value=True)
        else:
            use_sample_params = False
        
        omega_m = st.slider("Omega_m", 0.1, 0.5, 0.3, 0.01)
        sigma_8 = st.slider("sigma_8", 0.6, 1.0, 0.8, 0.01)
        
        st.markdown("---")
        st.subheader("Feedback")
        a_sn1 = st.slider("A_SN1 (stellar)", 0.25, 4.0, 1.0, 0.05)
        a_agn1 = st.slider("A_AGN1 (AGN)", 0.25, 4.0, 1.0, 0.05)
        a_sn2 = st.slider("A_SN2", 0.5, 2.0, 1.0, 0.05)
        a_agn2 = st.slider("A_AGN2", 0.5, 2.0, 1.0, 0.05)
        
        manual_params = {
            "omega_m": omega_m,
            "sigma_8": sigma_8,
            "a_sn1": a_sn1,
            "a_agn1": a_agn1,
            "a_sn2": a_sn2,
            "a_agn2": a_agn2,
        }
    
    if "Mcdm" not in test_data:
        st.error("No test data found!")
        st.stop()
    
    dm_maps = test_data["Mcdm"]
    n_samples = dm_maps.shape[0] if dm_maps.ndim == 3 else 1
    
    col1, col2 = st.columns([1, 4])
    with col1:
        sample_idx = st.number_input("Sample", 0, n_samples - 1, 0)
    
    if dm_maps.ndim == 3:
        input_map = dm_maps[sample_idx]
    else:
        input_map = dm_maps
    
    gt_map = None
    if "Mtot" in test_data:
        mtot_maps = test_data["Mtot"]
        gt_map = mtot_maps[sample_idx] if mtot_maps.ndim == 3 else mtot_maps
    
    if use_sample_params and has_lh_params and sample_idx < len(test_data["params"]):
        sp = test_data["params"][sample_idx]
        params = {
            "omega_m": float(sp[0]),
            "sigma_8": float(sp[1]),
            "a_sn1": float(sp[2]) if len(sp) > 2 else 1.0,
            "a_agn1": float(sp[3]) if len(sp) > 3 else 1.0,
            "a_sn2": float(sp[4]) if len(sp) > 4 else 1.0,
            "a_agn2": float(sp[5]) if len(sp) > 5 else 1.0,
        }
        st.info(f"Sample params: Om={params['omega_m']:.3f}, s8={params['sigma_8']:.3f}")
    else:
        params = manual_params
    
    if st.button("Run Inference", type="primary"):
        with st.spinner("Running..."):
            output_map = run_inference(model, device, input_map, params)
            diff_map = output_map - input_map
        
        st.session_state["input"] = input_map
        st.session_state["output"] = output_map
        st.session_state["diff"] = diff_map
        st.session_state["gt"] = gt_map
    
    if "input" in st.session_state:
        st.subheader("Results (interactive - scroll to zoom, drag to pan)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = create_heatmap(st.session_state["input"], "Input: Dark Matter",
                                 COLORMAPS[colormap], log_scale, dark_theme=dark_theme)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_heatmap(st.session_state["output"], "Predicted: Total Matter",
                                 COLORMAPS[colormap], log_scale, dark_theme=dark_theme)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            fig3 = create_heatmap(st.session_state["diff"], "Baryonic Effect",
                                 COLORMAPS[colormap], log_scale=False, diverging=True, 
                                 dark_theme=dark_theme)
            st.plotly_chart(fig3, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input", f"[{st.session_state['input'].min():.2e}, {st.session_state['input'].max():.2e}]")
        with col2:
            st.metric("Output", f"[{st.session_state['output'].min():.2e}, {st.session_state['output'].max():.2e}]")
        with col3:
            st.metric("Diff", f"[{st.session_state['diff'].min():.2e}, {st.session_state['diff'].max():.2e}]")
        
        if st.session_state["gt"] is not None:
            st.subheader("Ground Truth Comparison")
            
            error = st.session_state["output"] - st.session_state["gt"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_gt = create_heatmap(st.session_state["gt"], "Ground Truth",
                                       COLORMAPS[colormap], log_scale, dark_theme=dark_theme)
                st.plotly_chart(fig_gt, use_container_width=True)
            
            with col2:
                fig_pred = create_heatmap(st.session_state["output"], "Prediction",
                                         COLORMAPS[colormap], log_scale, dark_theme=dark_theme)
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col3:
                fig_err = create_heatmap(error, "Error",
                                        COLORMAPS[colormap], log_scale=False, diverging=True,
                                        dark_theme=dark_theme)
                st.plotly_chart(fig_err, use_container_width=True)
            
            mse = np.mean(error**2)
            mae = np.mean(np.abs(error))
            rel = np.mean(np.abs(error) / (np.abs(st.session_state["gt"]) + 1e-8))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.4e}")
            col2.metric("MAE", f"{mae:.4e}")
            col3.metric("Rel Error", f"{rel:.2%}")
    
    st.markdown("---")
    st.subheader("Parameter Sweep")
    
    col1, col2 = st.columns(2)
    with col1:
        sweep_param = st.selectbox("Parameter", ["a_sn1", "a_agn1", "omega_m", "sigma_8"])
    with col2:
        n_steps = st.slider("Steps", 2, 5, 3)
    
    if st.button("Run Sweep"):
        ranges = {
            "omega_m": (0.1, 0.5), "sigma_8": (0.6, 1.0),
            "a_sn1": (0.25, 4.0), "a_agn1": (0.25, 4.0),
            "a_sn2": (0.5, 2.0), "a_agn2": (0.5, 2.0),
        }
        
        values = np.linspace(*ranges[sweep_param], n_steps)
        
        cols = st.columns(n_steps)
        stats = []
        
        for i, val in enumerate(values):
            test_p = params.copy()
            test_p[sweep_param] = float(val)
            
            with st.spinner(f"Running {sweep_param}={val:.2f}..."):
                out = run_inference(model, device, input_map, test_p)
            
            with cols[i]:
                fig = create_heatmap(out, f"{sweep_param}={val:.2f}",
                                    COLORMAPS[colormap], log_scale, dark_theme=dark_theme)
                st.plotly_chart(fig, use_container_width=True)
            
            stats.append({"param": f"{val:.2f}", "mean": out.mean(), "std": out.std()})
        
        st.dataframe(stats)


if __name__ == "__main__":
    main()
