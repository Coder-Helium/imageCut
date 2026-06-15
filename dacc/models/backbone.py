from __future__ import annotations

import torch
from torch import nn


class TinyBackbone(nn.Module):
    """Small CNN backbone for reproducible smoke tests and ablations."""

    def __init__(self, in_channels: int = 3, width: int = 64, out_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._block(width, width * 2, stride=2)
        self.stage2 = self._block(width * 2, width * 4, stride=2)
        self.stage3 = self._block(width * 4, out_dim, stride=2)
        self.out_dim = out_dim

    def _block(self, c_in: int, c_out: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        pooled = feat.mean(dim=(2, 3))
        return feat, pooled

