#!/usr/bin/env python3
"""
FastAPI inference server for Bayronik baryonic field emulator.

Run: uv run uvicorn server:app --host 0.0.0.0 --port 8000
"""

import io
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model.ufno import UFNO2dConditional

app = FastAPI(title="Bayronik Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
DEVICE = None


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


@app.on_event("startup")
async def load_model():
    global MODEL, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    model_path = Path(__file__).parent / "weights" / "best_ufno_cond_LH_IllustrisTNG.pth"
    
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        return
    
    MODEL = UFNO2dConditional(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        modes=32,
        depth=4,
        num_conditions=6,
    )
    MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"Model loaded from {model_path}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    if MODEL is None:
        return {"error": "Model not loaded"}
    
    input_array = np.array(request.input_map, dtype=np.float32)
    h, w = input_array.shape
    
    print(f"Inference with params: Om={request.omega_m:.3f}, s8={request.sigma_8:.3f}, "
          f"ASN1={request.a_sn1:.3f}, AAGN1={request.a_agn1:.3f}, "
          f"ASN2={request.a_sn2:.3f}, AAGN2={request.a_agn2:.3f}")
    
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
    
    print(f"Input range: [{input_array.min():.3f}, {input_array.max():.3f}]")
    print(f"Output range: [{output_array.min():.3f}, {output_array.max():.3f}]")
    
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
        return {"error": "Model not loaded"}
    
    content = await file.read()
    input_array = np.load(io.BytesIO(content))
    
    if input_array.ndim == 2:
        input_array = input_array[np.newaxis, ...]
    
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
        
        results.append(output_tensor.squeeze().cpu().numpy())
    
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
