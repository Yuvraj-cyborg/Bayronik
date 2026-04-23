"""
Multi-scale and physics-informed loss functions for baryonic field emulation.

Includes:
- Pixel-wise losses (MSE, L1, Huber)
- Spectral losses (power spectrum matching)
- Higher-order statistics losses (skewness, kurtosis)
- Physics-informed losses (mass conservation, baryon fraction)
- Composite losses with learnable weights

These losses ensure the model captures both:
- Field-level accuracy (pixel-wise)
- Statistical accuracy (power spectrum, non-Gaussianity)
- Physical constraints (conservation laws)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PowerSpectrum2D:
    """
    Compute 2D power spectrum for density fields.
    
    Used for spectral loss computation.
    """
    
    def __init__(self, resolution: int = 256, device: str = "cpu"):
        self.resolution = resolution
        
        # Pre-compute k-bins
        kx = torch.fft.fftfreq(resolution, d=1.0/resolution)
        ky = torch.fft.fftfreq(resolution, d=1.0/resolution)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2)
        
        self.k = k.to(device)
        
        # Define k-bins (logarithmic spacing)
        self.k_bins = torch.logspace(0, torch.log10(torch.tensor(resolution/2)), 32).to(device)
        self.n_bins = len(self.k_bins) - 1

    def __call__(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute power spectrum P(k) for a batch of 2D fields.
        
        Args:
            field: (B, 1, H, W) density field
        
        Returns:
            k_centers: (n_bins,) k-bin centers
            power: (B, n_bins) power spectrum values
        """
        B = field.shape[0]
        
        # FFT
        field_fft = torch.fft.fft2(field.squeeze(1))
        power_2d = torch.abs(field_fft) ** 2
        
        # Bin averaging
        power = torch.zeros(B, self.n_bins, device=field.device)
        for i in range(self.n_bins):
            mask = (self.k >= self.k_bins[i]) & (self.k < self.k_bins[i+1])
            if mask.sum() > 0:
                power[:, i] = power_2d[:, mask].mean(dim=-1)
        
        # k-bin centers
        k_centers = (self.k_bins[:-1] + self.k_bins[1:]) / 2
        
        return k_centers, power


class PowerSpectrumLoss(nn.Module):
    """
    Loss based on power spectrum matching.
    
    Penalizes differences in the power spectrum P(k) between
    predicted and target fields. Can optionally weight different
    k-scales differently (e.g., emphasize small scales).
    
    Args:
        resolution: Field resolution
        log_space: Compute loss in log(P(k)) space
        scale_weights: Optional weights for different k-bins
    """
    
    def __init__(
        self,
        resolution: int = 256,
        log_space: bool = True,
        scale_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.log_space = log_space
        self.ps_computer = None
        self.resolution = resolution
        
        if scale_weights is not None:
            self.register_buffer('scale_weights', scale_weights)
        else:
            self.scale_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        
        # Lazy init PS computer with correct device
        if self.ps_computer is None:
            self.ps_computer = PowerSpectrum2D(self.resolution, device=device)
        
        # Compute power spectra
        _, ps_pred = self.ps_computer(pred)
        _, ps_target = self.ps_computer(target)
        
        # Avoid log(0)
        eps = 1e-10
        
        if self.log_space:
            ps_pred = torch.log(ps_pred + eps)
            ps_target = torch.log(ps_target + eps)
        
        # Compute loss
        diff = (ps_pred - ps_target) ** 2
        
        if self.scale_weights is not None:
            diff = diff * self.scale_weights.unsqueeze(0)
        
        return diff.mean()


class FieldStatisticsLoss(nn.Module):
    """
    Loss based on field statistics (mean, variance, skewness, kurtosis).
    
    Higher-order statistics (skewness, kurtosis) capture non-Gaussianity
    which is important for cosmological field emulation.
    """
    
    def __init__(
        self,
        use_mean: bool = True,
        use_variance: bool = True,
        use_skewness: bool = True,
        use_kurtosis: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.use_mean = use_mean
        self.use_variance = use_variance
        self.use_skewness = use_skewness
        self.use_kurtosis = use_kurtosis
        
        default_weights = {'mean': 1.0, 'variance': 1.0, 'skewness': 0.5, 'kurtosis': 0.5}
        self.weights = weights or default_weights

    def _compute_moments(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute statistical moments of the field."""
        # Flatten spatial dimensions
        x_flat = rearrange(x, 'b c h w -> b (c h w)')
        
        mean = x_flat.mean(dim=-1)
        centered = x_flat - mean.unsqueeze(-1)
        
        variance = (centered ** 2).mean(dim=-1)
        std = torch.sqrt(variance + 1e-10)
        
        # Standardize for higher moments
        standardized = centered / (std.unsqueeze(-1) + 1e-10)
        
        skewness = (standardized ** 3).mean(dim=-1)
        kurtosis = (standardized ** 4).mean(dim=-1) - 3  # Excess kurtosis
        
        return {
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_stats = self._compute_moments(pred)
        target_stats = self._compute_moments(target)
        
        loss = 0.0
        
        if self.use_mean:
            loss += self.weights['mean'] * F.mse_loss(pred_stats['mean'], target_stats['mean'])
        
        if self.use_variance:
            loss += self.weights['variance'] * F.mse_loss(pred_stats['variance'], target_stats['variance'])
        
        if self.use_skewness:
            loss += self.weights['skewness'] * F.mse_loss(pred_stats['skewness'], target_stats['skewness'])
        
        if self.use_kurtosis:
            loss += self.weights['kurtosis'] * F.mse_loss(pred_stats['kurtosis'], target_stats['kurtosis'])
        
        return loss


class GradientLoss(nn.Module):
    """
    Loss on spatial gradients to preserve edges and structure.
    
    Useful for preserving sharp features like halo cores
    and cosmic web filaments.
    """
    
    def __init__(self, order: int = 1):
        super().__init__()
        self.order = order
        
        # Sobel-like gradient kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def _gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial gradients."""
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = self._gradient(pred)
        target_gx, target_gy = self._gradient(target)
        
        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        
        if self.order >= 2:
            # Second order gradients (Laplacian-like)
            pred_gxx, _ = self._gradient(pred_gx)
            pred_gyy, _ = self._gradient(pred_gy)
            target_gxx, _ = self._gradient(target_gx)
            target_gyy, _ = self._gradient(target_gy)
            
            loss += 0.5 * (F.l1_loss(pred_gxx, target_gxx) + F.l1_loss(pred_gyy, target_gyy))
        
        return loss


class MassConservationLoss(nn.Module):
    """
    Physics-informed loss enforcing approximate mass conservation.
    
    Total mass should be conserved between input (DM) and output (total matter),
    with some expected baryon fraction added.
    """
    
    def __init__(self, expected_baryon_fraction: float = 0.157):
        """
        Args:
            expected_baryon_fraction: f_b = Omega_b / Omega_m ≈ 0.157 (Planck)
        """
        super().__init__()
        self.expected_baryon_fraction = expected_baryon_fraction

    def forward(
        self,
        pred: torch.Tensor,
        input_dm: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute mass conservation loss.
        
        In log-space (log1p), we expect:
            mean(pred) ≈ mean(input_dm) + log(1 + f_b)
        """
        pred_mean = pred.mean(dim=(1, 2, 3))
        input_mean = input_dm.mean(dim=(1, 2, 3))
        
        # Expected shift in log space
        expected_shift = torch.log(torch.tensor(1 + self.expected_baryon_fraction, device=pred.device))
        
        # Soft constraint: allow some deviation
        deviation = torch.abs(pred_mean - input_mean - expected_shift)
        
        return deviation.mean()


class MultiscaleLoss(nn.Module):
    """
    Multi-scale loss computed at different resolutions.
    
    Helps the model capture features at all scales:
    - Large scales: cosmic web, voids
    - Medium scales: halo profiles
    - Small scales: halo cores, substructure
    """
    
    def __init__(
        self,
        scales: Tuple[int, ...] = (1, 2, 4, 8),
        base_loss: str = "mse",
    ):
        super().__init__()
        self.scales = scales
        self.base_loss = F.mse_loss if base_loss == "mse" else F.l1_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                pred_scaled = F.avg_pool2d(pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
            
            # Weight smaller scales more (they're harder to get right)
            weight = 1.0 / scale
            total_loss += weight * self.base_loss(pred_scaled, target_scaled)
        
        # Normalize by total weight
        total_weight = sum(1.0 / s for s in self.scales)
        return total_loss / total_weight


class BaryonicEmulatorLoss(nn.Module):
    """
    Combined loss function for baryonic field emulation.
    
    Combines multiple loss terms:
    - Pixel-wise loss (MSE/L1)
    - Power spectrum loss (spectral matching)
    - Field statistics loss (higher-order moments)
    - Gradient loss (edge preservation)
    - Multi-scale loss (all scales)
    - Optional physics constraints
    
    Uses learnable or fixed weights for each component.
    
    Args:
        pixel_weight: Weight for pixel-wise MSE loss
        spectral_weight: Weight for power spectrum loss
        stats_weight: Weight for statistics loss
        gradient_weight: Weight for gradient loss
        multiscale_weight: Weight for multi-scale loss
        mass_weight: Weight for mass conservation
        resolution: Field resolution (for power spectrum)
        learnable_weights: If True, weights are learned during training
    """
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        spectral_weight: float = 0.1,
        stats_weight: float = 0.1,
        gradient_weight: float = 0.05,
        multiscale_weight: float = 0.1,
        mass_weight: float = 0.0,  # Set > 0 to enable
        resolution: int = 256,
        learnable_weights: bool = False,
    ):
        super().__init__()
        
        # Store weights
        if learnable_weights:
            # Learnable log-weights (softplus for positivity)
            self.log_weights = nn.Parameter(torch.tensor([
                pixel_weight, spectral_weight, stats_weight,
                gradient_weight, multiscale_weight, mass_weight
            ]).log())
        else:
            self.register_buffer('weights', torch.tensor([
                pixel_weight, spectral_weight, stats_weight,
                gradient_weight, multiscale_weight, mass_weight
            ]))
        
        self.learnable_weights = learnable_weights
        
        # Initialize loss components
        self.pixel_loss = nn.MSELoss()
        self.spectral_loss = PowerSpectrumLoss(resolution=resolution)
        self.stats_loss = FieldStatisticsLoss()
        self.gradient_loss = GradientLoss()
        self.multiscale_loss = MultiscaleLoss()
        self.mass_loss = MassConservationLoss() if mass_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_dm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            pred: Predicted field (B, 1, H, W)
            target: Target field (B, 1, H, W)
            input_dm: Input DM field for mass conservation (optional)
        
        Returns:
            Dictionary with individual losses and total
        """
        # Get weights
        if self.learnable_weights:
            weights = F.softplus(self.log_weights)
        else:
            weights = self.weights
        
        losses = {}
        
        # Pixel loss
        losses['pixel'] = self.pixel_loss(pred, target)
        
        # Spectral loss
        losses['spectral'] = self.spectral_loss(pred, target)
        
        # Statistics loss
        losses['stats'] = self.stats_loss(pred, target)
        
        # Gradient loss
        losses['gradient'] = self.gradient_loss(pred, target)
        
        # Multi-scale loss
        losses['multiscale'] = self.multiscale_loss(pred, target)
        
        # Mass conservation (if enabled and input provided)
        if self.mass_loss is not None and input_dm is not None:
            losses['mass'] = self.mass_loss(pred, input_dm, target)
        else:
            losses['mass'] = torch.tensor(0.0, device=pred.device)
        
        # Total weighted loss
        loss_values = torch.stack([
            losses['pixel'], losses['spectral'], losses['stats'],
            losses['gradient'], losses['multiscale'], losses['mass']
        ])
        
        losses['total'] = (weights * loss_values).sum()
        losses['weights'] = weights.detach()
        
        return losses


def create_loss(config: dict) -> BaryonicEmulatorLoss:
    """
    Factory function to create loss from config dict.
    
    Example config:
        {
            'pixel_weight': 1.0,
            'spectral_weight': 0.1,
            'stats_weight': 0.1,
            'gradient_weight': 0.05,
            'resolution': 256,
        }
    """
    return BaryonicEmulatorLoss(**config)


if __name__ == "__main__":
    # Test losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred = torch.randn(4, 1, 256, 256, device=device)
    target = torch.randn(4, 1, 256, 256, device=device)
    input_dm = torch.randn(4, 1, 256, 256, device=device)
    
    print("Testing loss functions on", device)
    print("-" * 50)
    
    # Test individual losses
    ps_loss = PowerSpectrumLoss(resolution=256)
    print(f"Power Spectrum Loss: {ps_loss(pred, target):.6f}")
    
    stats_loss = FieldStatisticsLoss()
    print(f"Statistics Loss: {stats_loss(pred, target):.6f}")
    
    grad_loss = GradientLoss()
    print(f"Gradient Loss: {grad_loss(pred, target):.6f}")
    
    ms_loss = MultiscaleLoss()
    print(f"Multiscale Loss: {ms_loss(pred, target):.6f}")
    
    print("-" * 50)
    
    # Test combined loss
    combined_loss = BaryonicEmulatorLoss(
        pixel_weight=1.0,
        spectral_weight=0.1,
        stats_weight=0.1,
        gradient_weight=0.05,
        multiscale_weight=0.1,
        mass_weight=0.01,
    )
    
    losses = combined_loss(pred, target, input_dm)
    print("Combined Loss Components:")
    for name, value in losses.items():
        if name == 'weights':
            continue
        print(f"  {name}: {value:.6f}")
