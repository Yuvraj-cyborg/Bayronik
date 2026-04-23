"""
Neural network architectures for baryonic feedback emulation.

Models:
    - UNet: Basic encoder-decoder with skip connections
    - ResUNet: UNet with residual blocks
    - AttentionUNet: UNet with self-attention in bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Basic conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class SelfAttention(nn.Module):
    """Self-attention block for capturing long-range dependencies."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class Down(nn.Module):
    """Downsampling: MaxPool -> ConvBlock"""
    
    def __init__(self, in_ch: int, out_ch: int, use_residual: bool = False):
        super().__init__()
        block = ResBlock if use_residual else ConvBlock
        self.pool = nn.MaxPool2d(2)
        self.conv = block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling: Upsample -> Concat -> ConvBlock"""
    
    def __init__(self, in_ch: int, out_ch: int, use_residual: bool = False):
        super().__init__()
        block = ResBlock if use_residual else ConvBlock
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for density field regression.
    
    Args:
        in_channels: Input channels (default: 1)
        out_channels: Output channels (default: 1)
        base_features: Base feature count, doubles each level (default: 64)
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features
        
        self.inc = ConvBlock(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 8)
        
        self.up1 = Up(f * 16, f * 4)
        self.up2 = Up(f * 8, f * 2)
        self.up3 = Up(f * 4, f)
        self.up4 = Up(f * 2, f)
        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ResUNet(nn.Module):
    """
    U-Net with residual blocks for better gradient flow.
    Recommended for deeper networks and larger datasets.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features
        
        self.inc = ResBlock(in_channels, f)
        self.down1 = Down(f, f * 2, use_residual=True)
        self.down2 = Down(f * 2, f * 4, use_residual=True)
        self.down3 = Down(f * 4, f * 8, use_residual=True)
        self.down4 = Down(f * 8, f * 8, use_residual=True)
        
        self.up1 = Up(f * 16, f * 4, use_residual=True)
        self.up2 = Up(f * 8, f * 2, use_residual=True)
        self.up3 = Up(f * 4, f, use_residual=True)
        self.up4 = Up(f * 2, f, use_residual=True)
        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class AttentionUNet(nn.Module):
    """
    U-Net with self-attention in bottleneck.
    Better at capturing large-scale correlations in density fields.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 64):
        super().__init__()
        f = base_features
        
        self.inc = ResBlock(in_channels, f)
        self.down1 = Down(f, f * 2, use_residual=True)
        self.down2 = Down(f * 2, f * 4, use_residual=True)
        self.down3 = Down(f * 4, f * 8, use_residual=True)
        self.down4 = Down(f * 8, f * 8, use_residual=True)
        
        self.attn = SelfAttention(f * 8)
        
        self.up1 = Up(f * 16, f * 4, use_residual=True)
        self.up2 = Up(f * 8, f * 2, use_residual=True)
        self.up3 = Up(f * 4, f, use_residual=True)
        self.up4 = Up(f * 2, f, use_residual=True)
        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.attn(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, Model in [("UNet", UNet), ("ResUNet", ResUNet), ("AttentionUNet", AttentionUNet)]:
        model = Model()
        x = torch.randn(2, 1, 256, 256)
        y = model(x)
        print(f"{name}: {count_parameters(model):,} params, output {y.shape}")
