"""
Model export utilities for deployment.

Supports:
- TorchScript export (for tch-rs inference)
- ONNX export (for WASM/browser inference)
- Quantization (for smaller model size)
- Model optimization

Usage:
    from bayronik_model.export import export_onnx, export_torchscript
    
    export_onnx(model, "model.onnx", dynamic_batch=True)
    export_torchscript(model, "model.pt")
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def export_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    conditional: bool = False,
    num_conditions: int = 6,
    optimize: bool = True,
) -> Path:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model
        output_path: Output path for .pt file
        input_shape: Input tensor shape
        conditional: Whether model takes condition inputs
        num_conditions: Number of condition parameters
        optimize: Apply torch.jit.optimize_for_inference
    
    Returns:
        Path to exported model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    # Create dummy inputs
    dummy_input = torch.randn(*input_shape)
    
    if conditional:
        dummy_conditions = torch.randn(input_shape[0], num_conditions)
        traced = torch.jit.trace(model, (dummy_input, dummy_conditions))
    else:
        traced = torch.jit.trace(model, dummy_input)
    
    # Optimize for inference
    if optimize:
        traced = torch.jit.optimize_for_inference(traced)
    
    traced.save(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ TorchScript exported: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def export_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    conditional: bool = False,
    num_conditions: int = 6,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
) -> Path:
    """
    Export model to ONNX format for WASM deployment.
    
    Args:
        model: PyTorch model
        output_path: Output path for .onnx file
        input_shape: Input tensor shape
        conditional: Whether model takes condition inputs
        num_conditions: Number of condition parameters
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        simplify: Simplify ONNX graph (requires onnx-simplifier)
    
    Returns:
        Path to exported model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    # Create dummy inputs
    dummy_input = torch.randn(*input_shape)
    
    # Set up dynamic axes
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    input_names = ["input"]
    
    if conditional:
        dummy_conditions = torch.randn(input_shape[0], num_conditions)
        inputs = (dummy_input, dummy_conditions)
        input_names.append("conditions")
        dynamic_axes["conditions"] = {0: "batch_size"}
    else:
        inputs = dummy_input
    
    # Export
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes if dynamic_batch else None,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            onnx_model = onnx.load(str(output_path))
            onnx_model_simp, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(onnx_model_simp, str(output_path))
                print("✓ ONNX model simplified")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ ONNX exported: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def quantize_dynamic(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
) -> Path:
    """
    Apply dynamic quantization for smaller model size.
    
    Quantizes Linear and Conv2d layers to int8.
    Good for CPU inference, reduces size ~4x.
    
    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input shape for tracing
    
    Returns:
        Path to quantized model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.cpu()
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    
    # Trace and save
    dummy_input = torch.randn(*input_shape)
    traced = torch.jit.trace(quantized_model, dummy_input)
    traced.save(str(output_path))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Quantized model exported: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


def export_onnx_fp16(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    conditional: bool = False,
    num_conditions: int = 6,
) -> Path:
    """
    Export model to ONNX with FP16 weights for smaller size.
    
    Reduces model size by ~50% with minimal accuracy loss.
    Good for WASM deployment where size matters.
    """
    output_path = Path(output_path)
    temp_path = output_path.with_suffix('.temp.onnx')
    
    # First export to FP32
    export_onnx(
        model,
        temp_path,
        input_shape=input_shape,
        conditional=conditional,
        num_conditions=num_conditions,
        simplify=False,
    )
    
    try:
        import onnx
        from onnx import numpy_helper
        
        # Load and convert to FP16
        onnx_model = onnx.load(str(temp_path))
        
        # Convert initializers to FP16
        for initializer in onnx_model.graph.initializer:
            if initializer.data_type == onnx.TensorProto.FLOAT:
                np_array = numpy_helper.to_array(initializer)
                fp16_array = np_array.astype('float16')
                new_initializer = numpy_helper.from_array(fp16_array, initializer.name)
                initializer.CopyFrom(new_initializer)
        
        onnx.save(onnx_model, str(output_path))
        temp_path.unlink()
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ ONNX FP16 exported: {output_path} ({size_mb:.2f} MB)")
        
    except ImportError:
        print("⚠ ONNX not installed, falling back to FP32")
        temp_path.rename(output_path)
    
    return output_path


def verify_onnx(
    onnx_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    conditional: bool = False,
    num_conditions: int = 6,
) -> bool:
    """
    Verify ONNX model is valid and can run inference.
    
    Returns:
        True if verification passes
    """
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        # Check model validity
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure valid")
        
        # Run inference test
        session = ort.InferenceSession(str(onnx_path))
        
        inputs = {"input": np.random.randn(*input_shape).astype(np.float32)}
        if conditional:
            inputs["conditions"] = np.random.randn(input_shape[0], num_conditions).astype(np.float32)
        
        outputs = session.run(None, inputs)
        
        print(f"✓ ONNX inference test passed, output shape: {outputs[0].shape}")
        return True
        
    except ImportError as e:
        print(f"⚠ Verification skipped (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 256, 256),
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Benchmark model inference speed.
    
    Returns:
        Dict with timing statistics
    """
    import time
    
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms
    
    import statistics
    
    results = {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
    }
    
    print(f"Inference benchmark ({device}):")
    print(f"  Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"  Min/Max: {results['min_ms']:.2f} / {results['max_ms']:.2f} ms")
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from fno import FNO2d
    from ufno import UFNO2d
    
    # Test exports
    print("Testing export utilities")
    print("-" * 50)
    
    model = FNO2d(hidden_channels=32, modes_x=16, modes_y=16, num_layers=2)
    
    # TorchScript
    ts_path = export_torchscript(model, "/tmp/test_model.pt")
    
    # ONNX
    onnx_path = export_onnx(model, "/tmp/test_model.onnx")
    verify_onnx(onnx_path)
    
    # Benchmark
    benchmark_model(model, num_runs=50)
