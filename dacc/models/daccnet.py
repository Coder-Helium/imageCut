from __future__ import annotations

import torch
from torch import nn

from dacc.box_ops import cxcywh_to_xyxy
from dacc.vocab import ACTION_VOCAB, ISSUE_VOCAB

from .backbone import TinyBackbone


class DACCNet(nn.Module):
    """Direction-aware crop generator with DETR-style crop queries."""

    def __init__(
        self,
        num_queries: int = 8,
        width: int = 64,
        hidden_dim: int = 256,
        num_actions: int = len(ACTION_VOCAB),
        num_issues: int = len(ISSUE_VOCAB),
        num_decoder_layers: int = 3,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.backbone = TinyBackbone(width=width, out_dim=hidden_dim)
        self.aspect_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.box_head = mlp(hidden_dim, hidden_dim, 3, 3)
        self.score_head = mlp(hidden_dim, hidden_dim, 1, 3)
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.issue_head = nn.Linear(hidden_dim, num_issues)

    def forward(self, image: torch.Tensor, aspect: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat, pooled = self.backbone(image)
        b, c, h, w = feat.shape
        memory = feat.flatten(2).transpose(1, 2)
        if aspect is None:
            aspect = torch.ones(b, 1, dtype=image.dtype, device=image.device)
        context = self.aspect_proj(aspect).unsqueeze(1)
        memory = memory + context
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        decoded = self.decoder(tgt=queries, memory=memory)
        raw_box = torch.sigmoid(self.box_head(decoded))
        # Predict cx, cy, scale. Width/height are derived from normalized target aspect.
        cxcy = raw_box[..., :2]
        scale = raw_box[..., 2:3].clamp(min=0.05, max=1.0)
        norm_aspect = aspect.clamp(min=1e-4, max=1e4).view(b, 1, 1)
        max_w = torch.where(norm_aspect >= 1.0, torch.ones_like(norm_aspect), norm_aspect)
        max_h = torch.where(norm_aspect >= 1.0, 1.0 / norm_aspect, torch.ones_like(norm_aspect))
        wh = torch.cat([scale * max_w, scale * max_h], dim=-1)
        boxes = cxcywh_to_xyxy(torch.cat([cxcy, wh], dim=-1))
        scores = torch.sigmoid(self.score_head(decoded)).squeeze(-1)
        return {
            "boxes": boxes,
            "scores": scores,
            "action_logits": self.action_head(decoded),
            "issue_logits": self.issue_head(decoded),
        }


def mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if i < num_layers - 1:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)
