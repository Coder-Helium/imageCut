from __future__ import annotations

import torch
from torch import nn

from .backbone import TinyBackbone


class CropRanker(nn.Module):
    """Candidate crop quality ranker.

    Inputs:
        image: full image tensor, Bx3xHxW
        crop: candidate crop tensor, Bx3xHxW
        box_feat: normalized box and teacher feature vector, Bx8
    Output:
        score in [0, 1], corresponding to original 1-5 score after scaling.
    """

    def __init__(self, width: int = 48, feat_dim: int = 192, box_feat_dim: int = 8) -> None:
        super().__init__()
        self.backbone = TinyBackbone(width=width, out_dim=feat_dim)
        self.box_mlp = nn.Sequential(
            nn.Linear(box_feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, crop: torch.Tensor, box_feat: torch.Tensor) -> torch.Tensor:
        _, full_vec = self.backbone(image)
        _, crop_vec = self.backbone(crop)
        box_vec = self.box_mlp(box_feat)
        logits = self.head(torch.cat([full_vec, crop_vec, box_vec], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

