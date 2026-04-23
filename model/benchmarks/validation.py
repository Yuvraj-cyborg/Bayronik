#!/usr/bin/env python3
"""Phase 2 scientific validation report for Bayronik.

Runs the trained conditional U-FNO across both CAMELS LH and CV splits and
emits three reproducible artifacts in ``reports/``:

* ``validation_report.json`` – every per-sample metric plus aggregates.
* ``validation_metrics.csv`` – flat CSV (one row per sample).
* ``validation_report.md`` – human-readable summary suitable for the public site.

Usage:
    cd model
    uv run python benchmarks/validation.py --lh-samples 32 --cv-samples 16
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch

from bayronik_model.ufno import UFNO2dConditional


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# CV simulations of IllustrisTNG share the fiducial cosmology of CAMELS.
# Numbers from camels_documentation: Omega_m=0.3, sigma_8=0.8, A_SN1=A_AGN1=A_SN2=A_AGN2=1.0
FIDUCIAL_COND = np.array([0.3, 0.8, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

# Indices into the conditioning vector (matches dataset.py order).
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]


@dataclass
class SampleMetrics:
    split: str
    idx: int
    infer_ms: float
    log_mse: float
    log_mae: float
    baseline_log_mse: float
    pk_pred_err: float
    pk_base_err: float
    cross_corr_mean: float
    cross_corr_min: float
    pdf_l1: float
    suppression_pred: float
    suppression_truth: float


@dataclass
class SplitSummary:
    split: str
    samples: int
    mean_infer_ms: float
    mean_log_mse: float
    mean_baseline_log_mse: float
    mse_improvement_x: float
    mean_log_mae: float
    mean_pk_log_mae: float
    mean_baseline_pk_log_mae: float
    pk_improvement_x: float
    mean_cross_corr: float
    p99_pdf_l1: float
    mean_suppression_err: float
    pk_bins: int
    per_sample: List[dict] = field(default_factory=list)
    pk_log_mae_by_bin: list = field(default_factory=list)
    pk_relative_err_by_bin: list = field(default_factory=list)


def load_model(device: torch.device) -> UFNO2dConditional:
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"trained weights not found at {WEIGHTS}. run training or scp the .pth file."
        )
    model = UFNO2dConditional(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        modes=32,
        depth=4,
        num_conditions=6,
    )
    state = torch.load(WEIGHTS, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _prep(field_map: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(np.asarray(field_map, dtype=np.float32), 0.0, None))


def _radial_bins(n: int, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the radial wavenumber grid and bin edges (cell units)."""
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kkx, kky = np.meshgrid(kx, ky, indexing="ij")
    kr = np.sqrt(kkx**2 + kky**2)
    edges = np.geomspace(1.0, kr.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return kr, edges, centers


def _bin_radial(values: np.ndarray, kr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (kr >= lo) & (kr < hi)
        if np.any(mask):
            out[i] = float(values[mask].mean())
        else:
            out[i] = np.nan
    return out


def power_and_cross(
    pred_field: np.ndarray, truth_field: np.ndarray, dm_field: np.ndarray, bins: int
) -> dict:
    """Return P_truth, P_pred, P_dm, r(k) per radial bin (cell units)."""
    n = pred_field.shape[0]
    kr, edges, centers = _radial_bins(n, bins)

    fp = np.fft.fft2(pred_field)
    ft = np.fft.fft2(truth_field)
    fd = np.fft.fft2(dm_field)
    norm = n * n

    pk_pred = (np.abs(fp) ** 2) / norm
    pk_truth = (np.abs(ft) ** 2) / norm
    pk_dm = (np.abs(fd) ** 2) / norm
    cross_pt = np.real(fp * np.conj(ft)) / norm

    def to_radial(arr):
        return _bin_radial(arr, kr, edges)

    pred_pk = to_radial(pk_pred)
    truth_pk = to_radial(pk_truth)
    dm_pk = to_radial(pk_dm)
    cross_pk = to_radial(cross_pt)

    denom = np.sqrt(np.clip(pred_pk * truth_pk, 1e-30, None))
    rk = np.where(denom > 0, cross_pk / denom, np.nan)

    return {
        "k": centers,
        "pk_truth": truth_pk,
        "pk_pred": pred_pk,
        "pk_dm": dm_pk,
        "rk": rk,
    }


def pdf_l1(pred: np.ndarray, truth: np.ndarray, bins: int = 80) -> float:
    rng = (
        float(min(pred.min(), truth.min())),
        float(max(pred.max(), truth.max())),
    )
    if rng[1] - rng[0] < 1e-12:
        return 0.0
    p_hist, _ = np.histogram(pred.ravel(), bins=bins, range=rng, density=True)
    t_hist, _ = np.histogram(truth.ravel(), bins=bins, range=rng, density=True)
    return float(0.5 * np.sum(np.abs(p_hist - t_hist)) * (rng[1] - rng[0]) / bins)


def suppression_ratio(field_baryon: np.ndarray, field_dm: np.ndarray) -> float:
    """Mean of P_baryon / P_dm over interior of the spectrum.

    Treated as a single scalar to flag strong over/under-suppression.
    """
    n = field_baryon.shape[0]
    kr, edges, _ = _radial_bins(n, 24)
    fb = np.fft.fft2(field_baryon)
    fd = np.fft.fft2(field_dm)
    norm = n * n
    pk_b = _bin_radial((np.abs(fb) ** 2) / norm, kr, edges)
    pk_d = _bin_radial((np.abs(fd) ** 2) / norm, kr, edges)
    ratio = pk_b / np.clip(pk_d, 1e-30, None)
    return float(np.nanmean(ratio[~np.isnan(ratio)]))


def predict_log(
    model: UFNO2dConditional, dm_log: np.ndarray, cond: np.ndarray, device: torch.device
) -> tuple[np.ndarray, float]:
    x = torch.from_numpy(dm_log).unsqueeze(0).unsqueeze(0).to(device)
    c = torch.from_numpy(cond).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(x, c).squeeze().detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred, (time.perf_counter() - start) * 1000.0


def run_split(
    split: str,
    dm_path: Path,
    mtot_path: Path,
    cond_for_idx,
    indices: Iterable[int],
    bins: int,
    model: UFNO2dConditional,
    device: torch.device,
) -> SplitSummary:
    dm = np.load(dm_path, mmap_mode="r")
    mtot = np.load(mtot_path, mmap_mode="r")

    samples: list[SampleMetrics] = []
    pk_log_mae_bins: list[np.ndarray] = []
    pk_rel_err_bins: list[np.ndarray] = []

    for idx in indices:
        dm_field = np.asarray(dm[idx], dtype=np.float32)
        truth_field = np.asarray(mtot[idx], dtype=np.float32)
        cond = cond_for_idx(idx).astype(np.float32)

        dm_log = _prep(dm_field)
        truth_log = _prep(truth_field)

        pred_log, infer_ms = predict_log(model, dm_log, cond, device)

        pk = power_and_cross(pred_log, truth_log, dm_log, bins=bins)

        pk_log_pred = np.log(np.clip(pk["pk_pred"], 1e-30, None))
        pk_log_truth = np.log(np.clip(pk["pk_truth"], 1e-30, None))
        pk_log_dm = np.log(np.clip(pk["pk_dm"], 1e-30, None))
        per_bin_log_mae = np.abs(pk_log_pred - pk_log_truth)
        per_bin_rel_err = np.abs(pk["pk_pred"] - pk["pk_truth"]) / np.clip(
            pk["pk_truth"], 1e-30, None
        )
        pk_log_mae_bins.append(per_bin_log_mae)
        pk_rel_err_bins.append(per_bin_rel_err)

        rk = pk["rk"]
        rk_finite = rk[np.isfinite(rk)]

        sup_pred = suppression_ratio(pred_log, dm_log)
        sup_truth = suppression_ratio(truth_log, dm_log)

        sample = SampleMetrics(
            split=split,
            idx=int(idx),
            infer_ms=float(infer_ms),
            log_mse=float(np.mean((pred_log - truth_log) ** 2)),
            log_mae=float(np.mean(np.abs(pred_log - truth_log))),
            baseline_log_mse=float(np.mean((dm_log - truth_log) ** 2)),
            pk_pred_err=float(np.nanmean(per_bin_log_mae)),
            pk_base_err=float(np.nanmean(np.abs(pk_log_dm - pk_log_truth))),
            cross_corr_mean=float(rk_finite.mean()) if rk_finite.size else float("nan"),
            cross_corr_min=float(rk_finite.min()) if rk_finite.size else float("nan"),
            pdf_l1=pdf_l1(pred_log, truth_log),
            suppression_pred=sup_pred,
            suppression_truth=sup_truth,
        )
        samples.append(sample)

    if not samples:
        raise RuntimeError(f"no samples produced for split={split}")

    pk_log_mae_arr = np.vstack(pk_log_mae_bins)
    pk_rel_err_arr = np.vstack(pk_rel_err_bins)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pk_log_mae_by_bin = np.nanmean(pk_log_mae_arr, axis=0)
        pk_relative_err_by_bin = np.nanmean(pk_rel_err_arr, axis=0)
    pk_log_mae_by_bin = np.nan_to_num(pk_log_mae_by_bin, nan=0.0).tolist()
    pk_relative_err_by_bin = np.nan_to_num(pk_relative_err_by_bin, nan=0.0).tolist()

    summary = SplitSummary(
        split=split,
        samples=len(samples),
        mean_infer_ms=float(np.mean([s.infer_ms for s in samples])),
        mean_log_mse=float(np.mean([s.log_mse for s in samples])),
        mean_baseline_log_mse=float(np.mean([s.baseline_log_mse for s in samples])),
        mse_improvement_x=float(
            np.mean([s.baseline_log_mse for s in samples])
            / max(np.mean([s.log_mse for s in samples]), 1e-30)
        ),
        mean_log_mae=float(np.mean([s.log_mae for s in samples])),
        mean_pk_log_mae=float(np.mean([s.pk_pred_err for s in samples])),
        mean_baseline_pk_log_mae=float(np.mean([s.pk_base_err for s in samples])),
        pk_improvement_x=float(
            np.mean([s.pk_base_err for s in samples])
            / max(np.mean([s.pk_pred_err for s in samples]), 1e-30)
        ),
        mean_cross_corr=float(np.nanmean([s.cross_corr_mean for s in samples])),
        p99_pdf_l1=float(np.percentile([s.pdf_l1 for s in samples], 99)),
        mean_suppression_err=float(
            np.mean([abs(s.suppression_pred - s.suppression_truth) for s in samples])
        ),
        pk_bins=pk_log_mae_arr.shape[1],
        per_sample=[asdict(s) for s in samples],
        pk_log_mae_by_bin=pk_log_mae_by_bin,
        pk_relative_err_by_bin=pk_relative_err_by_bin,
    )
    return summary


def parameter_sensitivity(
    model: UFNO2dConditional,
    base_field: np.ndarray,
    base_cond: np.ndarray,
    device: torch.device,
    steps: int = 5,
) -> list[dict]:
    """Sweep each conditioning parameter and record output norms.

    A trained model should respond smoothly and (for cosmology params) monotonically.
    We don't enforce any threshold here, only emit the trace for inspection.
    """
    dm_log = _prep(base_field)
    base_pred, _ = predict_log(model, dm_log, base_cond, device)
    base_norm = float(np.linalg.norm(base_pred))

    results = []
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = (0.1, 0.5) if name == "Omega_m" else (
            (0.6, 1.0) if name == "sigma_8" else (0.25, 4.0)
        )
        sweep = np.linspace(lo, hi, steps)
        norms = []
        log_mses = []
        for v in sweep:
            cond = base_cond.copy()
            cond[i] = float(v)
            pred, _ = predict_log(model, dm_log, cond, device)
            norms.append(float(np.linalg.norm(pred)))
            log_mses.append(float(np.mean((pred - base_pred) ** 2)))
        deltas = np.diff(norms)
        results.append(
            {
                "param": name,
                "values": sweep.tolist(),
                "output_norms": norms,
                "delta_log_mse_vs_base": log_mses,
                "monotonic_in_norm": bool(np.all(deltas >= -1e-3) or np.all(deltas <= 1e-3)),
                "norm_response": float(np.std(norms) / max(base_norm, 1e-12)),
            }
        )
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[SplitSummary], sensitivity: list[dict],
                   weights_sha: str, env: dict, args: argparse.Namespace) -> None:
    lines = ["# Bayronik validation report", ""]
    lines += [
        "| field | value |",
        "| --- | --- |",
        f"| weights | `{weights_sha[:16]}...` |",
        f"| device | `{env['device']}` |",
        f"| torch | {env['torch']} |",
        f"| numpy | {env['numpy']} |",
        f"| python | {env['python']} |",
        f"| platform | {env['platform']} |",
        "",
    ]
    lines += [
        "## Aggregate metrics",
        "",
        "| split | n | infer ms | log MSE | baseline | × | log MAE | P(k) MAE | base | × | r(k) | suppression err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.split} | {s.samples} | {s.mean_infer_ms:.1f} | "
            f"{s.mean_log_mse:.4f} | {s.mean_baseline_log_mse:.4f} | "
            f"{s.mse_improvement_x:.2f} | {s.mean_log_mae:.4f} | "
            f"{s.mean_pk_log_mae:.4f} | {s.mean_baseline_pk_log_mae:.4f} | "
            f"{s.pk_improvement_x:.2f} | {s.mean_cross_corr:.3f} | "
            f"{s.mean_suppression_err:.4f} |"
        )

    for s in summaries:
        lines += ["", f"### Per-bin P(k) MAE — {s.split}", ""]
        lines += ["| bin | log MAE | rel. err |", "| ---: | ---: | ---: |"]
        for i, (mae, rel) in enumerate(
            zip(s.pk_log_mae_by_bin, s.pk_relative_err_by_bin)
        ):
            lines.append(f"| {i} | {mae:.4f} | {rel:.4f} |")

    lines += ["", "## Parameter sensitivity (single sample)", ""]
    lines += ["| parameter | values | norm response | monotonic |", "| --- | --- | ---: | :---: |"]
    for s in sensitivity:
        vals = ", ".join(f"{v:.2f}" for v in s["values"])
        lines.append(
            f"| {s['param']} | {vals} | {s['norm_response']:.3e} | "
            f"{'yes' if s['monotonic_in_norm'] else 'no'} |"
        )

    lines += ["", f"_generated by `validation.py` ({args.lh_samples} LH + {args.cv_samples} CV samples, {args.bins} k-bins)_", ""]
    path.write_text("\n".join(lines))


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lh-samples", type=int, default=32)
    parser.add_argument("--cv-samples", type=int, default=16)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--report-dir",
        default=str(REPORTS),
        help="Output directory (default: model/reports)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny config for smoke-testing (4 LH + 2 CV samples).",
    )
    args = parser.parse_args()

    if args.quick:
        args.lh_samples = 4
        args.cv_samples = 2

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(device)

    weights_sha = sha256(WEIGHTS)
    env = {
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    params = np.loadtxt(DATA / "params_LH_IllustrisTNG.txt").astype(np.float32)
    dm_lh = np.load(DATA / "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy", mmap_mode="r")
    n_maps = dm_lh.shape[0]
    maps_per_sim = max(1, n_maps // len(params))

    def cond_lh(idx: int) -> np.ndarray:
        return params[min(idx // maps_per_sim, len(params) - 1), :6]

    def cond_cv(_idx: int) -> np.ndarray:
        return FIDUCIAL_COND.copy()

    lh_indices = np.linspace(0, n_maps - 1, args.lh_samples, dtype=int).tolist()
    dm_cv = np.load(DATA / "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy", mmap_mode="r")
    cv_indices = np.linspace(0, dm_cv.shape[0] - 1, args.cv_samples, dtype=int).tolist()

    summaries: list[SplitSummary] = []
    summaries.append(
        run_split(
            "LH",
            DATA / "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy",
            DATA / "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy",
            cond_lh,
            lh_indices,
            args.bins,
            model,
            device,
        )
    )
    summaries.append(
        run_split(
            "CV",
            DATA / "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy",
            DATA / "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy",
            cond_cv,
            cv_indices,
            args.bins,
            model,
            device,
        )
    )

    sample_field = np.asarray(dm_lh[lh_indices[0]], dtype=np.float32)
    sensitivity = parameter_sensitivity(
        model,
        sample_field,
        cond_lh(lh_indices[0]).astype(np.float32),
        device,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "validation_report.json"
    csv_path = report_dir / "validation_metrics.csv"
    md_path = report_dir / "validation_report.md"

    flat_rows: list[dict] = []
    for s in summaries:
        flat_rows.extend(s.per_sample)

    json_payload = {
        "weights_sha256": weights_sha,
        "env": env,
        "args": vars(args),
        "splits": [
            {
                **{k: v for k, v in asdict(s).items() if k not in {"per_sample"}},
                "per_sample": s.per_sample,
            }
            for s in summaries
        ],
        "parameter_sensitivity": sensitivity,
    }
    json_path.write_text(json.dumps(json_payload, indent=2))
    write_csv(csv_path, flat_rows)
    write_markdown(md_path, summaries, sensitivity, weights_sha, env, args)

    print("Bayronik validation report")
    print("--------------------------")
    print(f"weights sha256:  {weights_sha[:16]}...")
    print(f"device:          {device}")
    for s in summaries:
        print(
            f"[{s.split}] n={s.samples}  log MSE={s.mean_log_mse:.4f} "
            f"(×{s.mse_improvement_x:.2f})  P(k) MAE={s.mean_pk_log_mae:.4f} "
            f"(×{s.pk_improvement_x:.2f})  r(k)={s.mean_cross_corr:.3f}  "
            f"infer={s.mean_infer_ms:.1f}ms"
        )
    print(f"json: {json_path}")
    print(f"csv:  {csv_path}")
    print(f"md:   {md_path}")


if __name__ == "__main__":
    main()
