#!/usr/bin/env python3
"""Export trained model to TorchScript."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model import UNet, ResUNet, AttentionUNet


MODELS = {"unet": UNet, "resunet": ResUNet, "attention": AttentionUNet}


def export(weights_path: Path, output_path: Path, model_type: str):
    if not weights_path.exists():
        print(f"Error: {weights_path} not found")
        sys.exit(1)
    
    Model = MODELS[model_type]
    model = Model(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    traced = torch.jit.trace(model, torch.randn(1, 1, 256, 256))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    
    print(f"Exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet", choices=list(MODELS.keys()))
    
    args = parser.parse_args()
    export(Path(args.weights), Path(args.output), args.model)


if __name__ == "__main__":
    main()
