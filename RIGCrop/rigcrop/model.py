from __future__ import annotations

import torch
from torch import nn

from .box_ops import tensor_coverage, tensor_sanitize_xyxy
from .schema import ACTIONS, RELATION_POLICIES, ROLES


class TinyBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 48, out_dim: int = 192) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = _block(width, width * 2, stride=2)
        self.stage2 = _block(width * 2, width * 4, stride=2)
        self.stage3 = _block(width * 4, out_dim, stride=2)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return feat, feat.mean(dim=(2, 3))


def _block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


class RIGCropModel(nn.Module):
    """Image-only RIG-Crop student.

    The model predicts a latent entity/relation graph from the full image, then
    conditions each candidate crop score on graph-derived crop utility features.
    """

    def __init__(
        self,
        width: int = 48,
        feat_dim: int = 192,
        graph_dim: int = 128,
        max_nodes: int = 8,
        box_feat_dim: int = 8,
        num_roles: int = len(ROLES),
        num_relations: int = len(RELATION_POLICIES),
        num_actions: int = len(ACTIONS),
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.num_roles = num_roles
        self.num_relations = num_relations
        self.backbone = TinyBackbone(width=width, out_dim=feat_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(feat_dim, graph_dim * max_nodes),
            nn.GELU(),
            nn.Linear(graph_dim * max_nodes, graph_dim * max_nodes),
        )
        self.node_box = nn.Linear(graph_dim, 4)
        self.node_role = nn.Linear(graph_dim, num_roles)
        self.node_importance = nn.Linear(graph_dim, 1)
        self.node_valid = nn.Linear(graph_dim, 1)
        self.relation_head = nn.Sequential(
            nn.Linear(graph_dim * 4, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, num_relations),
        )
        self.relation_weight = nn.Sequential(
            nn.Linear(graph_dim * 4, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, 1),
        )
        self.action_head = nn.Sequential(nn.Linear(feat_dim, graph_dim), nn.GELU(), nn.Linear(graph_dim, num_actions))
        self.box_mlp = nn.Sequential(nn.Linear(box_feat_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.utility_head = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.score_head = nn.Sequential(
            nn.Linear(feat_dim * 2 + 64 + 5, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode_graph(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _, full_vec = self.backbone(image)
        node_tokens = self.graph_proj(full_vec).view(image.size(0), self.max_nodes, -1)
        node_boxes = tensor_sanitize_xyxy(torch.sigmoid(self.node_box(node_tokens)))
        node_role_logits = self.node_role(node_tokens)
        node_importance = torch.sigmoid(self.node_importance(node_tokens)).squeeze(-1)
        node_valid_logits = self.node_valid(node_tokens).squeeze(-1)
        rel_pair = _pair_features(node_tokens)
        relation_logits = self.relation_head(rel_pair)
        relation_weight = torch.sigmoid(self.relation_weight(rel_pair)).squeeze(-1)
        return {
            "full_vec": full_vec,
            "node_tokens": node_tokens,
            "node_boxes": node_boxes,
            "node_role_logits": node_role_logits,
            "node_importance": node_importance,
            "node_valid_logits": node_valid_logits,
            "relation_logits": relation_logits,
            "relation_weight": relation_weight,
            "action_logits": self.action_head(full_vec),
        }

    def forward(self, image: torch.Tensor, crop: torch.Tensor, box_feat: torch.Tensor, graph: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if graph is None:
            graph = self.encode_graph(image)
        _, crop_vec = self.backbone(crop)
        box_vec = self.box_mlp(box_feat)
        graph_feat = self.graph_features_for_crop(graph, box_feat[:, :4])
        utility = torch.sigmoid(self.utility_head(graph_feat)).squeeze(-1)
        logits = self.score_head(torch.cat([graph["full_vec"], crop_vec, box_vec, graph_feat], dim=-1)).squeeze(-1)
        out = dict(graph)
        out.update({"score": torch.sigmoid(logits), "utility": utility, "graph_feat": graph_feat})
        return out

    def graph_features_for_crop(self, graph: dict[str, torch.Tensor], crop_boxes: torch.Tensor) -> torch.Tensor:
        node_boxes = graph["node_boxes"]
        importance = graph["node_importance"]
        role_prob = graph["node_role_logits"].softmax(dim=-1)
        valid = torch.sigmoid(graph["node_valid_logits"])
        cov = tensor_coverage(node_boxes, crop_boxes) * valid
        non_distractor = 1.0 - role_prob[..., ROLES.index("distractor")]
        node_keep = (cov * importance * non_distractor).sum(dim=1)
        distractor = (cov * importance * role_prob[..., ROLES.index("distractor")]).sum(dim=1)
        boundary = (cov * (1.0 - cov) * 4.0 * importance * non_distractor).sum(dim=1)
        rel_prob = graph["relation_logits"].softmax(dim=-1)
        preserve_ids = [
            RELATION_POLICIES.index("preserve_relation"),
            RELATION_POLICIES.index("optional_preserve"),
            RELATION_POLICIES.index("avoid_cutting"),
            RELATION_POLICIES.index("leave_space"),
        ]
        preserve_prob = rel_prob[..., preserve_ids].sum(dim=-1)
        pair_cov = cov[:, :, None] * cov[:, None, :]
        rel_keep = (pair_cov * preserve_prob * graph["relation_weight"]).mean(dim=(1, 2))
        main_prob = role_prob[..., ROLES.index("main_subject")]
        main_cov = (cov * main_prob).sum(dim=1)
        return torch.stack([node_keep, rel_keep, distractor, boundary, main_cov], dim=-1)


def _pair_features(tokens: torch.Tensor) -> torch.Tensor:
    a = tokens[:, :, None, :].expand(-1, -1, tokens.size(1), -1)
    b = tokens[:, None, :, :].expand(-1, tokens.size(1), -1, -1)
    return torch.cat([a, b, a * b, (a - b).abs()], dim=-1)
