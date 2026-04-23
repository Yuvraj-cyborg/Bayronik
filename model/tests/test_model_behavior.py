from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from bayronik_model.ufno import UFNO2dConditional


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
DATA_DIR = ROOT / "data"
DM_LH = DATA_DIR / "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
MTOT_LH = DATA_DIR / "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
PARAMS_LH = DATA_DIR / "params_LH_IllustrisTNG.txt"


def _model() -> UFNO2dConditional:
    model = UFNO2dConditional(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        modes=32,
        depth=4,
        num_conditions=6,
    )
    return model


def _load_trained_model(device: torch.device) -> UFNO2dConditional:
    if not WEIGHTS.exists():
        pytest.skip(f"Model weights not found: {WEIGHTS}")

    model = _model()
    state = torch.load(WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _load_lh_sample(idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [p for p in [DM_LH, MTOT_LH, PARAMS_LH] if not p.exists()]
    if missing:
        pytest.skip(f"Missing CAMELS files: {missing}")

    dm = np.load(DM_LH, mmap_mode="r")
    mtot = np.load(MTOT_LH, mmap_mode="r")
    params = np.loadtxt(PARAMS_LH)

    maps_per_sim = max(1, dm.shape[0] // len(params))
    sim_idx = min(idx // maps_per_sim, len(params) - 1)

    input_map = np.array(dm[idx], dtype=np.float32)
    target_map = np.array(mtot[idx], dtype=np.float32)
    cond = params[sim_idx, :6].astype(np.float32)
    return input_map, target_map, cond


def _predict_log(
    model: UFNO2dConditional,
    input_map: np.ndarray,
    cond: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    safe = np.clip(input_map.astype(np.float32), 0, None)
    x = torch.from_numpy(np.log1p(safe)).unsqueeze(0).unsqueeze(0).to(device)
    c = torch.from_numpy(cond).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x, c).squeeze().cpu().numpy()
    return y.astype(np.float32)


def _power_spectrum(field: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    f = np.nan_to_num(field.astype(np.float64), copy=False)
    n = f.shape[0]
    fft = np.fft.fft2(f)
    power = np.abs(fft) ** 2 / (n * n)

    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kkx, kky = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kkx**2 + kky**2).ravel()
    pk = power.ravel()

    edges = np.linspace(1.0, kr.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.zeros_like(centers)
    for i in range(bins):
        mask = (kr >= edges[i]) & (kr < edges[i + 1])
        out[i] = pk[mask].mean() if np.any(mask) else np.nan
    good = np.isfinite(out) & (out > 0)
    return centers[good], out[good]


def test_ufno_conditional_forward_contract() -> None:
    torch.manual_seed(0)
    model = _model().eval()
    x = torch.randn(1, 1, 256, 256)
    cond = torch.tensor([[0.3, 0.8, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        out = model(x, cond)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_trained_model_output_is_finite_deterministic_and_nontrivial() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_trained_model(device)
    input_map, _target_map, cond = _load_lh_sample(0)

    y1 = _predict_log(model, input_map, cond, device)
    y2 = _predict_log(model, input_map, cond, device)

    assert y1.shape == (256, 256)
    assert np.isfinite(y1).all()
    assert np.allclose(y1, y2, rtol=1e-5, atol=1e-5)

    output_linear = np.expm1(y1)
    assert np.isfinite(output_linear).all()
    assert float(output_linear.max()) > float(output_linear.min())


@pytest.mark.slow
def test_trained_model_beats_dark_matter_identity_baseline() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_trained_model(device)
    input_map, target_map, cond = _load_lh_sample(0)

    pred_log = _predict_log(model, input_map, cond, device)
    input_log = np.log1p(np.clip(input_map, 0, None))
    target_log = np.log1p(np.clip(target_map, 0, None))

    baseline_mse = float(np.mean((input_log - target_log) ** 2))
    pred_mse = float(np.mean((pred_log - target_log) ** 2))

    assert pred_mse < baseline_mse, (
        f"Model should improve over raw DM baseline: pred_mse={pred_mse:.6f}, "
        f"baseline_mse={baseline_mse:.6f}"
    )


@pytest.mark.slow
def test_trained_model_matches_target_power_spectrum_better_than_baseline() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_trained_model(device)
    input_map, target_map, cond = _load_lh_sample(0)

    pred_log = _predict_log(model, input_map, cond, device)
    input_log = np.log1p(np.clip(input_map, 0, None))
    target_log = np.log1p(np.clip(target_map, 0, None))

    _k_t, pk_target = _power_spectrum(target_log)
    _k_p, pk_pred = _power_spectrum(pred_log)
    _k_i, pk_input = _power_spectrum(input_log)

    n = min(len(pk_target), len(pk_pred), len(pk_input))
    log_target = np.log(pk_target[:n])
    pred_err = float(np.mean(np.abs(np.log(pk_pred[:n]) - log_target)))
    baseline_err = float(np.mean(np.abs(np.log(pk_input[:n]) - log_target)))

    assert pred_err < baseline_err, (
        f"Predicted P(k) should be closer to target than DM baseline: "
        f"pred_err={pred_err:.6f}, baseline_err={baseline_err:.6f}"
    )
