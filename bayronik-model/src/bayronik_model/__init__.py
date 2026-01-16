"""
Bayronik Model - SOTA Baryonic feedback emulator for weak lensing cosmology.

This package provides neural operator models for mapping gravity-only
density fields to hydrodynamic fields, solving the "baryonic bottleneck"
in modern cosmology.

Models:
    Traditional CNN:
        - UNet: Basic encoder-decoder with skip connections
        - ResUNet: UNet with residual blocks
        - AttentionUNet: UNet with self-attention

    Fourier Neural Operators (SOTA):
        - FNO2d: 2D Fourier Neural Operator
        - FNO2dConditional: FNO with physics parameter conditioning
        - MultiscaleFNO2d: Multi-resolution FNO

    U-FNO (Hybrid - Recommended):
        - UFNO2d: U-Net + FNO hybrid
        - UFNO2dConditional: Conditional U-FNO
        - AttentionUFNO2d: U-FNO with attention

Losses:
    - BaryonicEmulatorLoss: Combined multi-scale loss
    - PowerSpectrumLoss: Spectral matching loss
    - FieldStatisticsLoss: Higher-order statistics loss

Data:
    - CAMELSDataset: CAMELS simulation dataset loader
    - create_dataloaders: Factory for train/val loaders

Export:
    - export_torchscript: Export for Rust inference
    - export_onnx: Export for WASM/browser inference

Example:
    >>> from bayronik_model import UFNO2d, BaryonicEmulatorLoss
    >>> model = UFNO2d(base_channels=32, modes=32, depth=4)
    >>> criterion = BaryonicEmulatorLoss()
    >>> pred = model(input_dm)
    >>> losses = criterion(pred, target)
"""

__version__ = "0.2.0"
__author__ = "Yuvraj Biswal"
__email__ = "yuvrajbiswalofficial@gmail.com"

# Traditional models
from .model import UNet, ResUNet, AttentionUNet

# FNO models
from .fno import (
    FNO2d,
    FNO2dConditional,
    MultiscaleFNO2d,
    SpectralConv2d,
    FNOBlock2d,
)

# U-FNO models
from .ufno import (
    UFNO2d,
    UFNO2dConditional,
    AttentionUFNO2d,
    UFNOEncoderBlock,
    UFNODecoderBlock,
)

# Loss functions
from .losses import (
    BaryonicEmulatorLoss,
    PowerSpectrumLoss,
    FieldStatisticsLoss,
    GradientLoss,
    MultiscaleLoss,
    MassConservationLoss,
    create_loss,
)

# Dataset
from .dataset import (
    CAMELSDataset,
    MultiSuiteDataset,
    MultiRedshiftDataset,
    create_dataloaders,
)

# Export utilities
from .export import (
    export_torchscript,
    export_onnx,
    export_onnx_fp16,
    quantize_dynamic,
    verify_onnx,
    benchmark_model,
)

# Convenience: all available models
MODELS = {
    # Traditional
    "unet": UNet,
    "resunet": ResUNet,
    "attention_unet": AttentionUNet,
    # FNO
    "fno": FNO2d,
    "fno_cond": FNO2dConditional,
    "fno_multiscale": MultiscaleFNO2d,
    # U-FNO
    "ufno": UFNO2d,
    "ufno_cond": UFNO2dConditional,
    "ufno_attention": AttentionUFNO2d,
}


def get_model(name: str, **kwargs):
    """
    Get model by name.
    
    Args:
        name: Model name (see MODELS dict)
        **kwargs: Model arguments
    
    Returns:
        Instantiated model
    
    Example:
        >>> model = get_model("ufno", base_channels=32, modes=32)
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)


__all__ = [
    # Version
    "__version__",
    # Traditional models
    "UNet",
    "ResUNet", 
    "AttentionUNet",
    # FNO models
    "FNO2d",
    "FNO2dConditional",
    "MultiscaleFNO2d",
    "SpectralConv2d",
    "FNOBlock2d",
    # U-FNO models
    "UFNO2d",
    "UFNO2dConditional",
    "AttentionUFNO2d",
    "UFNOEncoderBlock",
    "UFNODecoderBlock",
    # Losses
    "BaryonicEmulatorLoss",
    "PowerSpectrumLoss",
    "FieldStatisticsLoss",
    "GradientLoss",
    "MultiscaleLoss",
    "MassConservationLoss",
    "create_loss",
    # Dataset
    "CAMELSDataset",
    "MultiSuiteDataset",
    "MultiRedshiftDataset",
    "create_dataloaders",
    # Export
    "export_torchscript",
    "export_onnx",
    "export_onnx_fp16",
    "quantize_dynamic",
    "verify_onnx",
    "benchmark_model",
    # Utils
    "MODELS",
    "get_model",
]
