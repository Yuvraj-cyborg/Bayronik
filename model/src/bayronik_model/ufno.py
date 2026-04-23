"""
U-FNO: U-Net enhanced Fourier Neural Operator for baryonic field emulation.

Combines:
- U-Net's hierarchical encoder-decoder with skip connections
- FNO's spectral convolutions for global dependencies
- Attention mechanisms for improved feature aggregation

This architecture excels at capturing:
- Large-scale cosmic web structure (via FNO spectral convolutions)
- Small-scale halo features (via U-Net local convolutions)
- Multi-scale baryonic feedback effects (via skip connections)

Reference:
    Wen et al., "U-FNO—An enhanced Fourier neural operator-based deep-learning model 
    for multiphase flow"
    https://www.sciencedirect.com/science/article/pii/S0309170822000562
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .fno import SpectralConv2d


class UFNOEncoderBlock(nn.Module):
    """
    Encoder block combining spectral convolution with CNN.
    
    Architecture:
        Input -> SpectralConv + Conv -> LayerNorm -> GELU -> Conv -> Residual
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        downsample: bool = True,
    ):
        super().__init__()
        self.downsample = downsample
        
        # Spectral path
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes, modes)
        
        # Local path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection projection if channels change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Downsampling
        if downsample:
            self.pool = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
        
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - Downsampled output (for next encoder block)
            - Skip connection (for decoder)
        """
        # Combine spectral and local paths
        h = self.spectral_conv(x) + self.conv1(x)
        h = self.act(self.norm1(h))
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        
        # Residual
        skip = h + self.skip(x)
        
        # Downsample for next stage
        if self.downsample:
            out = self.pool(skip)
        else:
            out = skip
        
        return out, skip


class UFNODecoderBlock(nn.Module):
    """
    Decoder block with spectral convolution and skip connection fusion.
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        modes: int,
    ):
        super().__init__()
        
        # Upsample
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        
        # Fusion (concatenate skip + upsampled)
        fused_channels = in_channels + skip_channels
        
        # Spectral path
        self.spectral_conv = SpectralConv2d(fused_channels, out_channels, modes, modes)
        
        # Local path
        self.conv1 = nn.Conv2d(fused_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample and concatenate with skip
        h = self.upsample(x)
        
        # Handle size mismatch (if any)
        if h.shape[-2:] != skip.shape[-2:]:
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        h = torch.cat([h, skip], dim=1)
        
        # Spectral + local processing
        h = self.spectral_conv(h) + self.conv1(h)
        h = self.act(self.norm1(h))
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        
        return h


class UFNO2d(nn.Module):
    """
    U-FNO: U-Net enhanced Fourier Neural Operator.
    
    A powerful hybrid architecture that combines:
    - U-Net's encoder-decoder structure for multi-scale processing
    - FNO's spectral convolutions for capturing global correlations
    - Skip connections for preserving fine-grained features
    
    Particularly effective for:
    - Cosmological density field mapping
    - Multi-scale baryonic effects (AGN cores to cosmic web)
    - High-frequency detail preservation
    
    Args:
        in_channels: Input channels (1 for density)
        out_channels: Output channels (1 for density)
        base_channels: Base channel count (doubles each scale)
        modes: Base Fourier modes (halves each scale)
        depth: Number of encoder/decoder stages
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        modes: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        
        # Input projection
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        ch = base_channels
        m = modes
        for i in range(depth):
            out_ch = ch * 2 if i < depth - 1 else ch
            self.encoders.append(
                UFNOEncoderBlock(ch, out_ch, m, downsample=(i < depth - 1))
            )
            if i < depth - 1:
                ch = out_ch
                m = max(m // 2, 4)  # Don't go below 4 modes
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            SpectralConv2d(ch, ch, m, m),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.GELU(),
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            # Reverse order
            in_ch = ch
            skip_ch = ch  # Skip from encoder at same level
            out_ch = ch // 2 if i < depth - 2 else base_channels
            
            self.decoders.append(
                UFNODecoderBlock(in_ch, skip_ch, out_ch, m)
            )
            ch = out_ch
            m = min(m * 2, modes)
        
        # Output projection
        # After decoders: h has base_channels, skips[0] has base_channels*2
        self.project = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1),  # base + 2*base from skip[0]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(base_channels, out_channels, 1),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lift
        h = self.lift(x)
        
        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            h, skip = encoder(h)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder with skip connections (reverse order)
        for i, decoder in enumerate(self.decoders):
            skip_idx = -(i + 2)  # Skip corresponding encoder skip
            h = decoder(h, skips[skip_idx])
        
        # Final skip connection and projection
        h = torch.cat([h, skips[0]], dim=1)
        return self.project(h)


class UFNO2dConditional(nn.Module):
    """
    Conditional U-FNO with physics parameter injection.
    
    Extends UFNO2d with FiLM-based conditioning on:
    - Redshift (z)
    - Feedback parameters (A_AGN, A_SN)
    - Cosmological parameters (Omega_m, sigma_8)
    
    Uses per-layer FiLM projections to handle varying channel sizes.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        modes: int = 32,
        depth: int = 4,
        num_conditions: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.base_channels = base_channels
        
        # Input projection
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        
        # Build encoder and track channel sizes per layer
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        ch = base_channels
        m = modes
        encoder_channels = []  # Track output channels per encoder
        
        for i in range(depth):
            out_ch = ch * 2 if i < depth - 1 else ch
            encoder_channels.append(out_ch)
            self.encoders.append(
                UFNOEncoderBlock(ch, out_ch, m, downsample=(i < depth - 1))
            )
            self.encoder_norms.append(nn.GroupNorm(8, out_ch))
            if i < depth - 1:
                ch = out_ch
                m = max(m // 2, 4)
        
        self.encoder_channels = encoder_channels  # [64, 128, 256, 256] for depth=4, base=32
        
        # Shared condition embedding
        hidden_dim = base_channels * 4
        self.condition_embed = nn.Sequential(
            nn.Linear(num_conditions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Per-layer FiLM projections (gamma, beta for each layer's channel count)
        self.film_projections = nn.ModuleList([
            nn.Linear(hidden_dim, out_ch * 2)  # gamma and beta
            for out_ch in encoder_channels
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            SpectralConv2d(ch, ch, m, m),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.GELU(),
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            in_ch = ch
            skip_ch = ch
            out_ch = ch // 2 if i < depth - 2 else base_channels
            self.decoders.append(
                UFNODecoderBlock(in_ch, skip_ch, out_ch, m)
            )
            ch = out_ch
            m = min(m * 2, modes)
        
        # Output projection
        # After decoders: h has base_channels, skips[0] has base_channels*2
        self.project = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(base_channels, out_channels, 1),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize FiLM projections to identity (gamma=0 -> 1+gamma=1, beta=0)
        for proj in self.film_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode conditions to shared embedding
        if conditions is not None:
            cond_embed = self.condition_embed(conditions)  # (B, hidden_dim)
        else:
            cond_embed = None
        
        # Lift
        h = self.lift(x)
        
        # Encoder with per-layer FiLM conditioning
        skips = []
        for i, (encoder, norm, film_proj) in enumerate(
            zip(self.encoders, self.encoder_norms, self.film_projections)
        ):
            h, skip = encoder(h)
            
            # Apply FiLM conditioning
            if cond_embed is not None:
                film_params = film_proj(cond_embed)  # (B, out_ch * 2)
                ch = self.encoder_channels[i]
                gamma = film_params[:, :ch, None, None]      # (B, ch, 1, 1)
                beta = film_params[:, ch:, None, None]       # (B, ch, 1, 1)
                skip = (1 + gamma) * norm(skip) + beta
            
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder
        for i, decoder in enumerate(self.decoders):
            skip_idx = -(i + 2)
            h = decoder(h, skips[skip_idx])
        
        # Final projection
        h = torch.cat([h, skips[0]], dim=1)
        return self.project(h)


class AttentionUFNO2d(nn.Module):
    """
    U-FNO with self-attention for enhanced global reasoning.
    
    Applies multi-head self-attention at a reduced resolution (pooled)
    to capture long-range dependencies efficiently, then upsamples
    and fuses with the main features.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        modes: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        attn_resolution: int = 16,  # Reduced resolution for attention
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_resolution = attn_resolution
        
        # Main U-FNO backbone
        self.ufno = UFNO2d(
            in_channels=in_channels,
            out_channels=base_channels,  # Output intermediate features
            base_channels=base_channels,
            modes=modes,
            depth=depth,
            dropout=dropout,
        )
        
        # Self-attention branch at reduced resolution
        # Pool -> Attention -> Upsample
        self.attn_pool = nn.AdaptiveAvgPool2d(attn_resolution)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=base_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(base_channels)
        
        # Fusion: combine main features with attention-enhanced features
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        
        # Final projection
        self.final_project = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Get U-FNO features
        h = self.ufno(x)  # (B, base_channels, H, W)
        
        # Attention branch at reduced resolution
        h_pool = self.attn_pool(h)  # (B, base_channels, attn_res, attn_res)
        h_flat = rearrange(h_pool, 'b c h w -> b (h w) c')
        h_attn, _ = self.attention(h_flat, h_flat, h_flat)
        h_attn = self.attn_norm(h_attn + h_flat)  # Residual
        h_attn = rearrange(h_attn, 'b (h w) c -> b c h w', h=self.attn_resolution, w=self.attn_resolution)
        
        # Upsample attention features back to original resolution
        h_attn_up = F.interpolate(h_attn, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse main features with attention-enhanced features
        h_fused = self.fusion(torch.cat([h, h_attn_up], dim=1))
        
        return self.final_project(h_fused)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test U-FNO models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 256, 256, device=device)
    conditions = torch.randn(2, 5, device=device)
    
    print("Testing U-FNO architectures on", device)
    print("-" * 50)
    
    # Standard U-FNO
    model = UFNO2d(base_channels=32, modes=32, depth=4).to(device)
    y = model(x)
    print(f"UFNO2d: {count_parameters(model):,} params, output {y.shape}")
    
    # Conditional U-FNO
    model = UFNO2dConditional(base_channels=32, modes=32, depth=4).to(device)
    y = model(x, conditions)
    print(f"UFNO2dConditional: {count_parameters(model):,} params, output {y.shape}")
    
    # Attention U-FNO
    model = AttentionUFNO2d(base_channels=32, modes=32, depth=4).to(device)
    y = model(x)
    print(f"AttentionUFNO2d: {count_parameters(model):,} params, output {y.shape}")
