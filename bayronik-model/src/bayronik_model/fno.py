"""
Fourier Neural Operator (FNO) implementation for baryonic field emulation.

This module implements FNO and its variants optimized for cosmological field mapping:
- FNO2d: Standard 2D Fourier Neural Operator
- FNO2dConditional: FNO with conditional inputs (redshift, feedback params)

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations"
    https://arxiv.org/abs/2010.08895
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.
    
    Performs convolution in Fourier space by:
    1. FFT of input
    2. Multiply by learnable complex weights (truncated to modes_x, modes_y)
    3. Inverse FFT
    
    This captures global spatial dependencies efficiently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x  # Number of Fourier modes in x
        self.modes_y = modes_y  # Number of Fourier modes in y
        
        # Learnable complex weights for the Fourier modes
        # Scale factor for initialization (Xavier-like for complex)
        scale = 1 / (in_channels * out_channels)
        
        # Weights for different quadrants of the Fourier space
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication for batched inputs."""
        # x: (batch, in_ch, x, y), weights: (in_ch, out_ch, x, y)
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Prepare output tensor
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply relevant Fourier modes with learnable weights
        # Lower frequencies (positive wavenumbers)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.compl_mul2d(
            x_ft[:, :, :self.modes_x, :self.modes_y], self.weights1
        )
        # Higher frequencies (negative wavenumbers, stored at end due to FFT convention)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.compl_mul2d(
            x_ft[:, :, -self.modes_x:, :self.modes_y], self.weights2
        )
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2d(nn.Module):
    """
    Single FNO block combining spectral convolution with local convolution.
    
    Architecture:
        x -> SpectralConv -> + -> LayerNorm -> GELU -> out
             Conv1x1 -------^
    """
    
    def __init__(
        self,
        channels: int,
        modes_x: int,
        modes_y: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.conv = nn.Conv2d(channels, channels, 1)  # Local path
        
        # MLP for mixing
        hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, channels, 1),
            nn.Dropout(dropout),
        )
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral path + local path
        h = self.spectral_conv(x) + self.conv(x)
        # MLP with residual
        return x + self.mlp(self.norm(h))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for field-to-field mapping.
    
    Maps gravity-only density fields to hydrodynamic density fields.
    
    Architecture:
        Input -> Lift (project to high-dim) -> [FNOBlock] x N -> Project -> Output
    
    Args:
        in_channels: Input channels (default: 1 for density field)
        out_channels: Output channels (default: 1)
        hidden_channels: Width of FNO layers
        modes_x: Number of Fourier modes in x direction
        modes_y: Number of Fourier modes in y direction
        num_layers: Number of FNO blocks
        mlp_ratio: MLP expansion ratio in each block
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        modes_x: int = 32,
        modes_y: int = 32,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Lift: project input to high-dimensional space
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 1),
        )
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, modes_x, modes_y, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Project: map back to output space
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 1),
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        # Lift to high-dimensional space
        h = self.lift(x)
        
        # Apply FNO blocks
        for block in self.fno_blocks:
            h = block(h)
        
        # Project to output space
        return self.project(h)


class FNO2dConditional(nn.Module):
    """
    Conditional FNO with physics parameter injection.
    
    Allows conditioning on:
    - Redshift (z)
    - AGN feedback strength (A_AGN)
    - Supernova feedback strength (A_SN)
    - Cosmological parameters (Omega_m, sigma_8, etc.)
    
    Uses FiLM (Feature-wise Linear Modulation) for conditioning.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        modes_x: int = 32,
        modes_y: int = 32,
        num_layers: int = 4,
        num_conditions: int = 5,  # z, A_AGN, A_SN, Omega_m, sigma_8
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Condition encoder: map physics params to modulation parameters
        self.condition_encoder = nn.Sequential(
            nn.Linear(num_conditions, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2 * num_layers),  # gamma, beta per layer
        )
        
        # Lift
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 1),
        )
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, modes_x, modes_y, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norms for FiLM conditioning
        self.layer_norms = nn.ModuleList([
            nn.GroupNorm(8, hidden_channels) for _ in range(num_layers)
        ])
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, 1),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize FiLM to identity (gamma=1, beta=0)
        nn.init.zeros_(self.condition_encoder[-1].weight)
        nn.init.zeros_(self.condition_encoder[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional conditioning.
        
        Args:
            x: Input tensor (B, C, H, W)
            conditions: Physics parameters (B, num_conditions) or None
        
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        batch_size = x.shape[0]
        
        # Encode conditions to FiLM parameters
        if conditions is not None:
            film_params = self.condition_encoder(conditions)
            film_params = rearrange(
                film_params, 
                'b (l p) -> b l p', 
                l=self.num_layers, 
                p=self.hidden_channels * 2
            )
            gammas = film_params[..., :self.hidden_channels]  # (B, L, C)
            betas = film_params[..., self.hidden_channels:]   # (B, L, C)
        else:
            gammas = None
            betas = None
        
        # Lift
        h = self.lift(x)
        
        # Apply FNO blocks with FiLM conditioning
        for i, (block, norm) in enumerate(zip(self.fno_blocks, self.layer_norms)):
            h = block(h)
            
            if gammas is not None:
                # FiLM: h = gamma * h + beta
                gamma = gammas[:, i, :, None, None]  # (B, C, 1, 1)
                beta = betas[:, i, :, None, None]
                h = norm(h)
                h = (1 + gamma) * h + beta
        
        return self.project(h)


class MultiscaleFNO2d(nn.Module):
    """
    Multi-scale FNO that processes at different resolutions.
    
    Captures both large-scale (cosmic web) and small-scale (halo cores) features.
    Uses a U-Net-like architecture with FNO blocks at each scale.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 48,
        modes: int = 16,
        num_layers_per_scale: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Encoder path (downsampling)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        self.enc1 = nn.ModuleList([
            FNOBlock2d(hidden_channels, modes, modes, dropout=dropout)
            for _ in range(num_layers_per_scale)
        ])
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1)
        
        self.enc2 = nn.ModuleList([
            FNOBlock2d(hidden_channels * 2, modes // 2, modes // 2, dropout=dropout)
            for _ in range(num_layers_per_scale)
        ])
        self.down2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            FNOBlock2d(hidden_channels * 4, modes // 4, modes // 4, dropout=dropout)
            for _ in range(num_layers_per_scale)
        ])
        
        # Decoder path (upsampling)
        self.up2 = nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.ModuleList([
            FNOBlock2d(hidden_channels * 4, modes // 2, modes // 2, dropout=dropout)  # *4 due to skip
            for _ in range(num_layers_per_scale)
        ])
        self.reduce2 = nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 1)
        
        self.up1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1)
        self.dec1 = nn.ModuleList([
            FNOBlock2d(hidden_channels * 2, modes, modes, dropout=dropout)  # *2 due to skip
            for _ in range(num_layers_per_scale)
        ])
        self.reduce1 = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lift
        h = self.lift(x)
        
        # Encoder
        for block in self.enc1:
            h = block(h)
        skip1 = h
        h = self.down1(h)
        
        for block in self.enc2:
            h = block(h)
        skip2 = h
        h = self.down2(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h)
        
        # Decoder
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        for block in self.dec2:
            h = block(h)
        h = self.reduce2(h)
        
        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        for block in self.dec1:
            h = block(h)
        h = self.reduce1(h)
        
        return self.project(h)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test FNO models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 256, 256, device=device)
    conditions = torch.randn(2, 5, device=device)
    
    print("Testing FNO architectures on", device)
    print("-" * 50)
    
    # Standard FNO
    model = FNO2d(hidden_channels=64, modes_x=32, modes_y=32, num_layers=4).to(device)
    y = model(x)
    print(f"FNO2d: {count_parameters(model):,} params, output {y.shape}")
    
    # Conditional FNO
    model = FNO2dConditional(hidden_channels=64, modes_x=32, modes_y=32, num_layers=4).to(device)
    y = model(x, conditions)
    print(f"FNO2dConditional: {count_parameters(model):,} params, output {y.shape}")
    
    # Multiscale FNO
    model = MultiscaleFNO2d(hidden_channels=48, modes=16).to(device)
    y = model(x)
    print(f"MultiscaleFNO2d: {count_parameters(model):,} params, output {y.shape}")
