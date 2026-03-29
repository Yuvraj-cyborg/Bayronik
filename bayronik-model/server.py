#!/usr/bin/env python3
"""
FastAPI inference server for Bayronik baryonic field emulator.

Run: make server
"""

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model.ufno import UFNO2dConditional

MODEL = None
DEVICE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global MODEL, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    model_path = Path(__file__).parent / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
    
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
    else:
        MODEL = UFNO2dConditional(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            modes=32,
            depth=4,
            num_conditions=6,
        )
        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        MODEL.to(DEVICE)
        MODEL.eval()
        print(f"Model loaded from {model_path}")
    
    yield


app = FastAPI(title="Bayronik Inference API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    input_map: list[list[float]]
    omega_m: float = 0.3
    sigma_8: float = 0.8
    a_sn1: float = 1.0
    a_agn1: float = 1.0
    a_sn2: float = 1.0
    a_agn2: float = 1.0


class InferenceResponse(BaseModel):
    output_map: list[list[float]]
    input_shape: list[int]
    output_shape: list[int]


CAMELS_DATA: dict = {}


def _load_camels():
    """Lazy-load CAMELS data as memory-mapped arrays."""
    if CAMELS_DATA:
        return
    data_dir = Path(__file__).parent / "data"
    for dt in ["LH", "CV"]:
        dm = data_dir / f"Maps_Mcdm_IllustrisTNG_{dt}_z=0.00.npy"
        mt = data_dir / f"Maps_Mtot_IllustrisTNG_{dt}_z=0.00.npy"
        if dm.exists() and "Mcdm" not in CAMELS_DATA:
            CAMELS_DATA["Mcdm"] = np.load(dm, mmap_mode="r")
            CAMELS_DATA["dataset_type"] = dt
        if mt.exists() and "Mtot" not in CAMELS_DATA:
            CAMELS_DATA["Mtot"] = np.load(mt, mmap_mode="r")
    for name in ["params_LH_IllustrisTNG.txt", "params_IllustrisTNG_LH.txt"]:
        p = data_dir / name
        if p.exists():
            CAMELS_DATA["params"] = np.loadtxt(p)
            break
    print(f"CAMELS data loaded: {list(CAMELS_DATA.keys())}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/dataset/info")
async def dataset_info():
    _load_camels()
    if "Mcdm" not in CAMELS_DATA:
        raise HTTPException(status_code=404, detail="No CAMELS data found")
    dm = CAMELS_DATA["Mcdm"]
    return {
        "dataset_type": CAMELS_DATA.get("dataset_type", "unknown"),
        "n_samples": int(dm.shape[0]),
        "resolution": int(dm.shape[1]),
    }


@app.get("/sample/{idx}")
async def get_sample(idx: int):
    _load_camels()
    dm = CAMELS_DATA.get("Mcdm")
    mt = CAMELS_DATA.get("Mtot")
    if dm is None:
        raise HTTPException(status_code=404, detail="No CAMELS data")
    if idx < 0 or idx >= dm.shape[0]:
        raise HTTPException(status_code=400, detail=f"Index {idx} out of range [0, {dm.shape[0]})")

    input_map = np.array(dm[idx], dtype=np.float32)
    gt_map = np.array(mt[idx], dtype=np.float32) if mt is not None else np.zeros_like(input_map)

    params_arr = CAMELS_DATA.get("params")
    if params_arr is not None:
        n_sims = len(params_arr)
        maps_per_sim = max(1, dm.shape[0] // n_sims)
        sim_idx = min(idx // maps_per_sim, n_sims - 1)
        params = params_arr[sim_idx].tolist()
    else:
        params = [0.3, 0.8, 1.0, 1.0, 1.0, 1.0]

    return {
        "input_map": input_map.tolist(),
        "ground_truth": gt_map.tolist(),
        "params": params,
    }


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_array = np.array(request.input_map, dtype=np.float32)
    h, w = input_array.shape
    
    print(f"Inference: Om={request.omega_m:.3f}, s8={request.sigma_8:.3f}, "
          f"ASN1={request.a_sn1:.3f}, AAGN1={request.a_agn1:.3f}")
    
    input_log = np.log1p(input_array)
    input_tensor = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    conditions = torch.tensor([
        [request.omega_m, request.sigma_8, request.a_sn1, 
         request.a_agn1, request.a_sn2, request.a_agn2]
    ], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        output_tensor = MODEL(input_tensor, conditions)
    
    output_log = output_tensor.squeeze().cpu().numpy()
    output_array = np.expm1(output_log)
    
    print(f"Input: [{input_array.min():.3f}, {input_array.max():.3f}] -> "
          f"Output: [{output_array.min():.3f}, {output_array.max():.3f}]")
    
    return InferenceResponse(
        output_map=output_array.tolist(),
        input_shape=[h, w],
        output_shape=list(output_array.shape),
    )


@app.post("/infer_npy")
async def infer_npy(
    file: UploadFile = File(...),
    omega_m: float = 0.3,
    sigma_8: float = 0.8,
    a_sn1: float = 1.0,
    a_agn1: float = 1.0,
    a_sn2: float = 1.0,
    a_agn2: float = 1.0,
):
    """Accept .npy file upload, return .npy output."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    content = await file.read()
    try:
        input_array = np.load(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy file: {e}")
    
    if input_array.ndim == 2:
        input_array = input_array[np.newaxis, ...]
    
    if input_array.ndim != 3 or input_array.shape[1] != 256 or input_array.shape[2] != 256:
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (N, 256, 256), got {input_array.shape}",
        )
    
    results = []
    for i in range(input_array.shape[0]):
        single_input = input_array[i]
        input_log = np.log1p(single_input.astype(np.float32))
        input_tensor = torch.from_numpy(input_log).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        conditions = torch.tensor([
            [omega_m, sigma_8, a_sn1, a_agn1, a_sn2, a_agn2]
        ], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            output_tensor = MODEL(input_tensor, conditions)
        
        output_log = output_tensor.squeeze().cpu().numpy()
        results.append(np.expm1(output_log))
    
    output_array = np.stack(results) if len(results) > 1 else results[0]
    
    buffer = io.BytesIO()
    np.save(buffer, output_array)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=output.npy"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
