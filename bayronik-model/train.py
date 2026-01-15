#!/usr/bin/env python3
"""
Training script for baryonic feedback emulation.

Usage:
    python train.py --dataset CV --epochs 20
    python train.py --dataset LH --model attention --mmap --epochs 50
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bayronik_model import UNet, ResUNet, AttentionUNet, CAMELSDataset


MODELS = {
    "unet": UNet,
    "resunet": ResUNet,
    "attention": AttentionUNet,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = get_device()
    print(f"Device: {device}")
    
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    dataset = CAMELSDataset(
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        mmap=args.mmap,
    )
    
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset: {args.dataset} ({train_size} train, {val_size} val)")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    
    # Model
    Model = MODELS[args.model]
    model = Model(in_channels=1, out_channels=1).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({num_params:,} params)")
    
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from: {args.resume}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_val_loss = float("inf")
    model_path = weights_dir / f"best_{args.model}_{args.dataset}.pth"
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                val_loss += criterion(model(inp), tgt).item()
        val_loss /= len(val_loader)
        
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.cpu()
            torch.save(model.state_dict(), model_path)
            model.to(device)
            improved = " *"
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {lr:.2e}{improved}")
    
    # Export
    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Weights: {model_path}")
    
    model.cpu()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    traced = torch.jit.trace(model, torch.randn(1, 1, 256, 256))
    ts_path = weights_dir / f"traced_{args.model}_{args.dataset}.pt"
    traced.save(str(ts_path))
    print(f"TorchScript: {ts_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CV", choices=["CV", "LH"])
    parser.add_argument("--model", type=str, default="unet", choices=list(MODELS.keys()))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--weights-dir", type=str, default="weights")
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    if sys.platform == "darwin":
        args.workers = 0
    
    train(args)


if __name__ == "__main__":
    main()
