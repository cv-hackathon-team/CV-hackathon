"""DepthGuardNet: a small U-Net with separate RGB and depth encoders.

Inputs are RGB, LR depth already upsampled with bicubic interpolation, and a validity
mask, so one model handles every scale factor. At each level, depth features gate the
RGB features through a sigmoid attention map, so RGB texture cannot dominate. A shared
decoder feeds three heads: a depth residual added to the input, a log-variance and
boundary logits.

The log-variance is constrained to be non-negative, so confidence = exp(-log_variance)
stays in (0, 1]. It is measured relative to a reference scale set in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels: int) -> nn.GroupNorm:
    """Return group normalisation, which works at the small batch sizes used here."""
    return nn.GroupNorm(8, channels)


@dataclass
class DepthGuardOutput:
    """Network outputs, each shaped (B, 1, H, W)."""

    depth: torch.Tensor
    log_variance: torch.Tensor
    confidence: torch.Tensor
    boundary_logits: torch.Tensor


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with normalisation and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create the block's layers."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        return self.activation(self.layers(x) + self.skip(x))


class Encoder(nn.Module):
    """Encoder that halves resolution between levels and returns each level's features."""

    def __init__(self, in_channels: int, channels: list[int], blocks_per_level: int) -> None:
        """Create the encoder levels."""
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.levels = nn.ModuleList()
        previous = channels[0]
        for level_channels in channels:
            blocks = [ResidualBlock(previous, level_channels)]
            blocks += [ResidualBlock(level_channels, level_channels) for _ in range(blocks_per_level - 1)]
            self.levels.append(nn.Sequential(*blocks))
            previous = level_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return features ordered from highest to lowest resolution."""
        x = self.stem(x)
        features = []
        for index, level in enumerate(self.levels):
            if index:
                x = F.avg_pool2d(x, 2)
            x = level(x)
            features.append(x)
        return features


class FusionBlock(nn.Module):
    """Gate RGB features with a sigmoid map computed from depth features only."""

    def __init__(self, channels: int) -> None:
        """Create the attention layers."""
        super().__init__()
        hidden = max(channels // 4, 8)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """Return depth features plus the gated RGB features."""
        return depth_features + self.attention(depth_features) * rgb_features


class Decoder(nn.Module):
    """Upsampling decoder with skip connections from the fused features."""

    def __init__(self, channels: list[int], blocks_per_level: int) -> None:
        """Create the decoder levels, from lowest resolution to highest."""
        super().__init__()
        self.levels = nn.ModuleList()
        reversed_channels = channels[::-1]
        for deeper, level_channels in zip(reversed_channels, reversed_channels[1:]):
            blocks = [ResidualBlock(deeper + level_channels, level_channels)]
            blocks += [ResidualBlock(level_channels, level_channels) for _ in range(blocks_per_level - 1)]
            self.levels.append(nn.Sequential(*blocks))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Return shared features at full resolution."""
        x = features[-1]
        for level, skip in zip(self.levels, features[-2::-1]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = level(torch.cat([x, skip], dim=1))
        return x


class DepthGuardNet(nn.Module):
    """Predict depth, confidence and boundaries from RGB and upsampled degraded depth."""

    def __init__(self, channels: list[int] | None = None, blocks_per_level: int = 1) -> None:
        """Create the encoders, fusion blocks, decoder and three heads."""
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        self.downsample_factor = 2 ** (len(channels) - 1)
        self.rgb_encoder = Encoder(3, channels, blocks_per_level)
        self.depth_encoder = Encoder(2, channels, blocks_per_level)
        self.fusion = nn.ModuleList(FusionBlock(c) for c in channels)
        self.decoder = Decoder(channels, blocks_per_level)
        self.depth_head = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.log_variance_head = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.boundary_head = nn.Conv2d(channels[0], 1, 3, padding=1)
        # Start as the identity on depth, so training begins from the bicubic input.
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> DepthGuardOutput:
        """Run the network. Height and width must be divisible by ``downsample_factor``.

        Depth is divided by each sample's mean valid depth internally, so the network
        sees similar values across scenes. The residual is scaled back to metres.
        """
        height, width = depth.shape[-2:]
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError(f"Size {(height, width)} is not divisible by {self.downsample_factor}")
        mask = validity_mask.to(depth.dtype)
        scale = ((depth * mask).sum((1, 2, 3), keepdim=True) / mask.sum((1, 2, 3), keepdim=True).clamp_min(1)).clamp_min(1e-3)
        rgb_features = self.rgb_encoder(rgb - 0.5)
        depth_features = self.depth_encoder(torch.cat([depth / scale - 1, mask], dim=1))
        fused = [block(r, d) for block, r, d in zip(self.fusion, rgb_features, depth_features)]
        shared = self.decoder(fused)
        log_variance = F.softplus(self.log_variance_head(shared))
        return DepthGuardOutput(
            depth=(depth + self.depth_head(shared) * scale).clamp_min(1e-3),
            log_variance=log_variance,
            confidence=torch.exp(-log_variance),
            boundary_logits=self.boundary_head(shared),
        )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
