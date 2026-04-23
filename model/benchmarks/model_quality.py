#!/usr/bin/env python3
"""Benchmark Bayronik model quality on CAMELS maps.

This is intentionally lightweight and dependency-free beyond the project stack.
It reports field-level and spectral metrics against the raw dark matter baseline.

Usage:
    cd model
    uv run python benchmarks/model_quality.py --samples 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bayronik_model.ufno import UFNO2dConditional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
DEFAULT_DATA = ROOT / "data"


def load_model(weights: Path, device: torch.device) -> UFNO2dConditional:
    model = UFNO2dConditional(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        modes=32,
        depth=4,
        num_conditions=6,
    )
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def power_spectrum(field: np.ndarray, bins: int = 48) -> np.ndarray:
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
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (kr >= lo) & (kr < hi)
        if np.any(mask):
            out.append(pk[mask].mean())
    return np.asarray(out, dtype=np.float64)


def run(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    weights = Path(args.weights)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    dm = np.load(data_dir / "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy", mmap_mode="r")
    mtot = np.load(data_dir / "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy", mmap_mode="r")
    params = np.loadtxt(data_dir / "params_LH_IllustrisTNG.txt").astype(np.float32)
    maps_per_sim = max(1, dm.shape[0] // len(params))

    model = load_model(weights, device)

    indices = np.linspace(0, min(dm.shape[0] - 1, args.max_index), args.samples, dtype=int)

    metrics = []
    start_all = time.perf_counter()
    for idx in indices:
        input_map = np.asarray(dm[idx], dtype=np.float32)
        target_map = np.asarray(mtot[idx], dtype=np.float32)
        cond = params[min(idx // maps_per_sim, len(params) - 1), :6]

        input_log = np.log1p(np.clip(input_map, 0, None))
        target_log = np.log1p(np.clip(target_map, 0, None))

        x = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(device)
        c = torch.from_numpy(cond).unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred_log = model(x, c).squeeze().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - start) * 1000.0

        baseline_mse = float(np.mean((input_log - target_log) ** 2))
        pred_mse = float(np.mean((pred_log - target_log) ** 2))
        pred_mae = float(np.mean(np.abs(pred_log - target_log)))

        pk_target = power_spectrum(target_log)
        pk_pred = power_spectrum(pred_log)
        pk_input = power_spectrum(input_log)
        n = min(len(pk_target), len(pk_pred), len(pk_input))
        pk_pred_err = float(np.mean(np.abs(np.log(pk_pred[:n]) - np.log(pk_target[:n]))))
        pk_base_err = float(np.mean(np.abs(np.log(pk_input[:n]) - np.log(pk_target[:n]))))

        metrics.append(
            {
                "idx": int(idx),
                "infer_ms": infer_ms,
                "log_mse": pred_mse,
                "log_mae": pred_mae,
                "baseline_log_mse": baseline_mse,
                "mse_improvement": baseline_mse / max(pred_mse, 1e-12),
                "pk_log_mae": pk_pred_err,
                "baseline_pk_log_mae": pk_base_err,
                "pk_improvement": pk_base_err / max(pk_pred_err, 1e-12),
            }
        )

    elapsed = time.perf_counter() - start_all
    summary = {
        "device": str(device),
        "weights": str(weights),
        "samples": len(metrics),
        "elapsed_s": elapsed,
        "mean_infer_ms": float(np.mean([m["infer_ms"] for m in metrics])),
        "mean_log_mse": float(np.mean([m["log_mse"] for m in metrics])),
        "mean_baseline_log_mse": float(np.mean([m["baseline_log_mse"] for m in metrics])),
        "mean_mse_improvement": float(np.mean([m["mse_improvement"] for m in metrics])),
        "mean_pk_log_mae": float(np.mean([m["pk_log_mae"] for m in metrics])),
        "mean_baseline_pk_log_mae": float(np.mean([m["baseline_pk_log_mae"] for m in metrics])),
        "mean_pk_improvement": float(np.mean([m["pk_improvement"] for m in metrics])),
        "per_sample": metrics,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-index", type=int, default=14999)
    parser.add_argument("--device", default=None, help="cpu, cuda, or leave empty for auto")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Bayronik Model Quality Benchmark")
    print("--------------------------------")
    print(f"Device:              {summary['device']}")
    print(f"Samples:             {summary['samples']}")
    print(f"Mean inference:      {summary['mean_infer_ms']:.2f} ms/map")
    print(f"Mean log MSE:        {summary['mean_log_mse']:.6f}")
    print(f"Baseline log MSE:    {summary['mean_baseline_log_mse']:.6f}")
    print(f"MSE improvement:     {summary['mean_mse_improvement']:.2f}x")
    print(f"Mean P(k) log MAE:   {summary['mean_pk_log_mae']:.6f}")
    print(f"Baseline P(k) MAE:   {summary['mean_baseline_pk_log_mae']:.6f}")
    print(f"P(k) improvement:    {summary['mean_pk_improvement']:.2f}x")


if __name__ == "__main__":
    main()
