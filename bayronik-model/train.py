#!/usr/bin/env python3
"""
Training script for Bayronik baryonic field emulator.

Features:
- FNO / U-FNO / traditional architectures
- Multi-scale loss functions (pixel + spectral + stats)
- Mixed precision training (AMP)
- Cosine annealing with warmup
- Gradient clipping
- Wandb integration
- Proper checkpointing and early stopping
- Validation on multiple metrics

Usage:
    uv run train.py --model fno --dataset LH --epochs 100
    uv run train.py --model ufno --dataset LH --epochs 100 --wandb
    uv run train.py --model ufno_cond --conditional --wandb
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bayronik_model import UNet, ResUNet, AttentionUNet
from bayronik_model.fno import FNO2d, FNO2dConditional, MultiscaleFNO2d
from bayronik_model.ufno import UFNO2d, UFNO2dConditional, AttentionUFNO2d
from bayronik_model.losses import BaryonicEmulatorLoss, PowerSpectrumLoss
from bayronik_model.dataset import CAMELSDataset, create_dataloaders


# Model registry
MODELS = {
    # Traditional architectures
    "unet": UNet,
    "resunet": ResUNet,
    "attention_unet": AttentionUNet,
    # FNO architectures
    "fno": FNO2d,
    "fno_cond": FNO2dConditional,
    "fno_multiscale": MultiscaleFNO2d,
    # U-FNO architectures
    "ufno": UFNO2d,
    "ufno_cond": UFNO2dConditional,
    "ufno_attention": AttentionUFNO2d,
}

# Default hyperparameters per model
MODEL_DEFAULTS = {
    "unet": {"base_features": 64},
    "resunet": {"base_features": 64},
    "attention_unet": {"base_features": 64},
    "fno": {"hidden_channels": 64, "modes_x": 32, "modes_y": 32, "num_layers": 4},
    "fno_cond": {"hidden_channels": 64, "modes_x": 32, "modes_y": 32, "num_layers": 4},
    "fno_multiscale": {"hidden_channels": 48, "modes": 16},
    "ufno": {"base_channels": 32, "modes": 32, "depth": 4},
    "ufno_cond": {"base_channels": 32, "modes": 32, "depth": 4},
    "ufno_attention": {"base_channels": 32, "modes": 32, "depth": 4},
}


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    model_name: str,
    conditional: bool = False,
    num_conditions: int = 6,
) -> nn.Module:
    """Create model from name."""
    # Handle conditional variants
    if conditional and "_cond" not in model_name:
        model_name = model_name + "_cond"
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    Model = MODELS[model_name]
    kwargs = MODEL_DEFAULTS.get(model_name, {}).copy()
    
    # Add conditional params
    if "cond" in model_name:
        kwargs["num_conditions"] = num_conditions
    
    return Model(in_channels=1, out_channels=1, **kwargs)


def create_scheduler(
    optimizer,
    num_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
):
    """Create learning rate scheduler with warmup."""
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    
    # Cosine annealing
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    
    # Combine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    
    return scheduler


def train_epoch(
    model: nn.Module,
    loader,
    criterion: BaryonicEmulatorLoss,
    optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    conditional: bool = False,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        # Unpack batch
        if conditional and len(batch) == 3:
            inp, tgt, params = batch
            inp, tgt, params = inp.to(device), tgt.to(device), params.to(device)
        else:
            inp, tgt = batch[:2]
            inp, tgt = inp.to(device), tgt.to(device)
            params = None
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast(enabled=scaler is not None):
            if params is not None:
                pred = model(inp, params)
            else:
                pred = model(inp)
            
            losses = criterion(pred, tgt, inp)
            loss = losses['total']
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for key, value in losses.items():
            if key not in ('total', 'weights'):
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
        
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # Average losses
    avg_losses = {'total': total_loss / num_batches}
    for key, value in loss_components.items():
        avg_losses[key] = value / num_batches
    
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: BaryonicEmulatorLoss,
    device: torch.device,
    conditional: bool = False,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    loss_components = {}
    num_batches = 0
    
    for batch in tqdm(loader, desc="Val", leave=False):
        if conditional and len(batch) == 3:
            inp, tgt, params = batch
            inp, tgt, params = inp.to(device), tgt.to(device), params.to(device)
        else:
            inp, tgt = batch[:2]
            inp, tgt = inp.to(device), tgt.to(device)
            params = None
        
        if params is not None:
            pred = model(inp, params)
        else:
            pred = model(inp)
        
        losses = criterion(pred, tgt, inp)
        
        total_loss += losses['total'].item()
        for key, value in losses.items():
            if key not in ('total', 'weights'):
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
        
        num_batches += 1
    
    avg_losses = {'total': total_loss / num_batches}
    for key, value in loss_components.items():
        avg_losses[key] = value / num_batches
    
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    path: Path,
    config: dict,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'config': config,
    }
    torch.save(checkpoint, path)


def train(args):
    """Main training function."""
    device = get_device()
    print(f"Device: {device}")
    
    # Paths
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.wandb:
        import wandb
        wandb.init(
            project="bayronik",
            name=f"{args.model}_{args.dataset}",
            config=vars(args),
        )
    
    # Data
    conditional = args.conditional or "cond" in args.model
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        suite=args.suite,
        dataset_type=args.dataset,
        augment_train=args.augment,
        return_params=conditional,
    )
    
    print(f"Dataset: {args.dataset} ({len(train_loader.dataset)} train, {len(val_loader.dataset)} val)")
    
    # Model
    model = create_model(args.model, conditional=conditional).to(device)
    num_params = count_parameters(model)
    print(f"Model: {args.model} ({num_params:,} params)")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_loss']
        print(f"Resumed from epoch {start_epoch}")
    
    # Loss function
    criterion = BaryonicEmulatorLoss(
        pixel_weight=args.pixel_weight,
        spectral_weight=args.spectral_weight,
        stats_weight=args.stats_weight,
        gradient_weight=args.gradient_weight,
        multiscale_weight=args.multiscale_weight,
        mass_weight=args.mass_weight,
        resolution=256,
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    
    if args.resume:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Mixed precision
    use_amp = device.type == "cuda" and args.amp
    scaler = GradScaler() if use_amp else None
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    
    # Training config for saving
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'suite': args.suite,
        'conditional': conditional,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_params': num_params,
    }
    
    # Paths for saving
    model_name = f"{args.model}_{args.dataset}_{args.suite}"
    best_path = weights_dir / f"best_{model_name}.pth"
    checkpoint_path = weights_dir / f"checkpoint_{model_name}.pth"
    
    # Training loop
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, conditional, args.grad_clip
        )
        
        # Validate
        val_losses = validate(model, val_loader, criterion, device, conditional)
        
        # Step scheduler
        scheduler.step()
        
        # Time
        epoch_time = time.time() - epoch_start
        
        # Check improvement
        improved = val_losses['total'] < best_val_loss
        if improved:
            best_val_loss = val_losses['total']
            patience_counter = 0
            
            # Save best model
            model.cpu()
            torch.save(model.state_dict(), best_path)
            model.to(device)
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_val_loss,
            checkpoint_path, config
        )
        
        # Current LR
        lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        marker = "*" if improved else ""
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} │ "
            f"train {train_losses['total']:.5f} │ "
            f"val {val_losses['total']:.5f} │ "
            f"lr {lr:.2e} │ "
            f"{epoch_time:.1f}s {marker}"
        )
        
        # Detailed losses
        if args.verbose:
            print(f"  └─ pixel: {val_losses.get('pixel', 0):.5f}, "
                  f"spectral: {val_losses.get('spectral', 0):.5f}, "
                  f"stats: {val_losses.get('stats', 0):.5f}")
        
        # Wandb logging
        if args.wandb:
            log_dict = {
                'epoch': epoch,
                'train/total': train_losses['total'],
                'val/total': val_losses['total'],
                'lr': lr,
            }
            for key in ['pixel', 'spectral', 'stats', 'gradient', 'multiscale']:
                if key in train_losses:
                    log_dict[f'train/{key}'] = train_losses[key]
                if key in val_losses:
                    log_dict[f'val/{key}'] = val_losses[key]
            wandb.log(log_dict)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {best_path}")
    print(f"{'='*60}")
    
    # Export to TorchScript
    print("\nExporting to TorchScript...")
    model.cpu()
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    # Trace
    dummy_input = torch.randn(1, 1, 256, 256)
    if conditional:
        dummy_params = torch.randn(1, 6)
        traced = torch.jit.trace(model, (dummy_input, dummy_params))
    else:
        traced = torch.jit.trace(model, dummy_input)
    
    ts_path = weights_dir / f"traced_{model_name}.pt"
    traced.save(str(ts_path))
    print(f"TorchScript saved to: {ts_path}")
    
    if args.wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train Bayronik baryonic field emulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="CV", choices=["CV", "LH"], help="Dataset type")
    parser.add_argument("--suite", type=str, default="IllustrisTNG", help="Simulation suite")
    
    # Model
    parser.add_argument("--model", type=str, default="ufno", choices=list(MODELS.keys()), help="Model architecture")
    parser.add_argument("--conditional", action="store_true", help="Use conditional model with physics params")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Loss weights
    parser.add_argument("--pixel-weight", type=float, default=1.0, help="Pixel loss weight")
    parser.add_argument("--spectral-weight", type=float, default=0.5, help="Spectral loss weight")
    parser.add_argument("--stats-weight", type=float, default=0.1, help="Statistics loss weight")
    parser.add_argument("--gradient-weight", type=float, default=0.05, help="Gradient loss weight")
    parser.add_argument("--multiscale-weight", type=float, default=0.1, help="Multiscale loss weight")
    parser.add_argument("--mass-weight", type=float, default=0.01, help="Mass conservation loss weight")
    
    # System
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--weights-dir", type=str, default="weights", help="Weights directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--augment", action="store_true", default=True, help="Use data augmentation")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Platform-specific adjustments
    if sys.platform == "darwin":
        args.workers = 0  # macOS fork issues
        args.amp = False  # MPS doesn't fully support AMP
    
    train(args)


if __name__ == "__main__":
    main()
