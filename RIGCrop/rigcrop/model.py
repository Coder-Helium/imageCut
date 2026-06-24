from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .backbones import BackboneOutput, build_visual_backbone
from .box_ops import tensor_coverage, tensor_sanitize_xyxy
from .schema import ACTIONS, RELATION_POLICIES, ROLES


class RIGCropModel(nn.Module):
    """RIGFormer: VLM-distilled graph transformer for image-only cropping.

    The public class name is kept as ``RIGCropModel`` so the existing training
    and inference scripts continue to work, but the architecture is now a
    transformerized relation-importance graph cropper:

    image tokens -> entity graph decoder -> relation head
                 -> graph-aware crop decision transformer -> crop score.
    """

    def __init__(
        self,
        width: int = 64,
        feat_dim: int = 256,
        graph_dim: int | None = None,
        d_model: int | None = None,
        max_nodes: int = 12,
        box_feat_dim: int = 8,
        backbone: Dict[str, Any] | None = None,
        num_entity_layers: int = 4,
        num_crop_layers: int = 2,
        num_crop_queries: int = 16,
        nhead: int = 8,
        dropout: float = 0.1,
        num_roles: int = len(ROLES),
        num_relations: int = len(RELATION_POLICIES),
        num_actions: int = len(ACTIONS),
    ) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.num_roles = num_roles
        self.num_relations = num_relations
        self.num_crop_queries = num_crop_queries
        self.d_model = int(d_model or graph_dim or feat_dim)

        heads = _valid_num_heads(self.d_model, nhead)
        self.backbone = build_visual_backbone(backbone, output_dim=self.d_model, fallback_width=width)

        self.entity_queries = nn.Parameter(torch.randn(max_nodes, self.d_model) * 0.02)
        entity_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_decoder = nn.TransformerDecoder(entity_layer, num_layers=max(1, num_entity_layers))
        self.entity_norm = nn.LayerNorm(self.d_model)

        self.node_box = _mlp(self.d_model, self.d_model, 4, 3, dropout=dropout)
        self.node_role = _mlp(self.d_model, self.d_model, num_roles, 2, dropout=dropout)
        self.node_importance = _mlp(self.d_model, self.d_model, 1, 2, dropout=dropout)
        self.node_valid = _mlp(self.d_model, self.d_model, 1, 2, dropout=dropout)

        self.geometry_proj = nn.Sequential(nn.Linear(8, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
        rel_dim = self.d_model * 5
        self.relation_head = _mlp(rel_dim, self.d_model, num_relations, 3, dropout=dropout)
        self.relation_weight = _mlp(rel_dim, self.d_model, 1, 3, dropout=dropout)
        self.action_head = _mlp(self.d_model, self.d_model, num_actions, 3, dropout=dropout)

        self.box_mlp = nn.Sequential(
            nn.Linear(box_feat_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.utility_component_proj = nn.Sequential(nn.Linear(5, self.d_model), nn.GELU(), nn.LayerNorm(self.d_model))
        self.crop_query_norm = nn.LayerNorm(self.d_model)
        self.crop_graph_layers = nn.ModuleList(
            [_CropGraphAttentionBlock(self.d_model, heads, dropout=dropout) for _ in range(max(1, num_crop_layers))]
        )
        self.crop_relation_gate = nn.Sequential(nn.Linear(5, self.d_model), nn.Sigmoid())
        self.utility_head = _mlp(self.d_model + 5, self.d_model, 1, 3, dropout=dropout)
        self.score_head = _mlp(self.d_model + 5, self.d_model, 1, 3, dropout=dropout)

        if num_crop_queries > 0:
            self.crop_queries = nn.Parameter(torch.randn(num_crop_queries, self.d_model) * 0.02)
            query_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=heads,
                dim_feedforward=self.d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.crop_query_decoder = nn.TransformerDecoder(query_layer, num_layers=2)
            self.query_box_head = _mlp(self.d_model, self.d_model, 4, 3, dropout=dropout)
            self.query_score_head = _mlp(self.d_model, self.d_model, 1, 3, dropout=dropout)
        else:
            self.crop_queries = None
            self.crop_query_decoder = None
            self.query_box_head = None
            self.query_score_head = None

    def encode_graph(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        visual = self.backbone(image)
        node_tokens = self._decode_entities(visual)
        node_boxes = tensor_sanitize_xyxy(torch.sigmoid(self.node_box(node_tokens)))
        node_role_logits = self.node_role(node_tokens)
        node_importance = torch.sigmoid(self.node_importance(node_tokens)).squeeze(-1)
        node_valid_logits = self.node_valid(node_tokens).squeeze(-1)
        rel_pair = self._relation_pair_features(node_tokens, node_boxes)
        relation_logits = self.relation_head(rel_pair)
        relation_weight = torch.sigmoid(self.relation_weight(rel_pair)).squeeze(-1)
        out = {
            "full_vec": visual.pooled,
            "node_tokens": node_tokens,
            "node_boxes": node_boxes,
            "node_role_logits": node_role_logits,
            "node_importance": node_importance,
            "node_valid_logits": node_valid_logits,
            "relation_logits": relation_logits,
            "relation_weight": relation_weight,
            "action_logits": self.action_head(visual.pooled),
        }
        if self.crop_queries is not None:
            q = self.crop_queries.unsqueeze(0).expand(image.size(0), -1, -1)
            query_tokens = self.crop_query_decoder(q, visual.tokens)
            out["query_tokens"] = query_tokens
            out["query_boxes"] = tensor_sanitize_xyxy(torch.sigmoid(self.query_box_head(query_tokens)))
            out["query_scores"] = torch.sigmoid(self.query_score_head(query_tokens)).squeeze(-1)
        return out

    def forward(
        self,
        image: torch.Tensor,
        crop: torch.Tensor,
        box_feat: torch.Tensor,
        graph: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if graph is None:
            graph = self.encode_graph(image)
        crop_visual = self.backbone(crop)
        graph_feat = self.graph_features_for_crop(graph, box_feat[:, :4])
        crop_token = crop_visual.pooled + self.box_mlp(box_feat) + self.utility_component_proj(graph_feat)
        crop_token = self.crop_query_norm(crop_token).unsqueeze(1)
        node_tokens = graph["node_tokens"]
        node_mask = torch.sigmoid(graph["node_valid_logits"]).detach() < 0.05
        if node_mask.all(dim=1).any():
            node_mask = node_mask.clone()
            node_mask[node_mask.all(dim=1), 0] = False
        gate = self.crop_relation_gate(graph_feat).unsqueeze(1)
        for layer in self.crop_graph_layers:
            crop_token = layer(crop_token, node_tokens, key_padding_mask=node_mask)
            crop_token = crop_token * (1.0 + gate)
        state = crop_token.squeeze(1)
        head_in = torch.cat([state, graph_feat], dim=-1)
        utility = torch.sigmoid(self.utility_head(head_in)).squeeze(-1)
        score = torch.sigmoid(self.score_head(head_in)).squeeze(-1)
        out = dict(graph)
        out.update({"score": score, "utility": utility, "graph_feat": graph_feat, "crop_state": state})
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

    def _decode_entities(self, visual: BackboneOutput) -> torch.Tensor:
        queries = self.entity_queries.unsqueeze(0).expand(visual.tokens.size(0), -1, -1)
        return self.entity_norm(self.entity_decoder(queries, visual.tokens))

    def _relation_pair_features(self, tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        a = tokens[:, :, None, :].expand(-1, -1, tokens.size(1), -1)
        b = tokens[:, None, :, :].expand(-1, tokens.size(1), -1, -1)
        geom = self.geometry_proj(_pair_geometry(boxes))
        return torch.cat([a, b, a * b, (a - b).abs(), geom], dim=-1)


class _CropGraphAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, crop_token: torch.Tensor, node_tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(crop_token)
        attended, _ = self.cross_attn(x, node_tokens, node_tokens, key_padding_mask=key_padding_mask, need_weights=False)
        crop_token = crop_token + attended
        crop_token = crop_token + self.ffn(self.norm2(crop_token))
        return crop_token


def _pair_geometry(boxes: torch.Tensor) -> torch.Tensor:
    boxes = tensor_sanitize_xyxy(boxes)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    ci = torch.stack([cx, cy, w, h], dim=-1)
    a = ci[:, :, None, :].expand(-1, -1, boxes.size(1), -1)
    b = ci[:, None, :, :].expand(-1, boxes.size(1), -1, -1)
    dxdy = a[..., :2] - b[..., :2]
    log_wh = torch.log(a[..., 2:] / b[..., 2:].clamp(min=1e-6)).clamp(min=-5.0, max=5.0)
    inter_x1 = torch.maximum(boxes[:, :, None, 0], boxes[:, None, :, 0])
    inter_y1 = torch.maximum(boxes[:, :, None, 1], boxes[:, None, :, 1])
    inter_x2 = torch.minimum(boxes[:, :, None, 2], boxes[:, None, :, 2])
    inter_y2 = torch.minimum(boxes[:, :, None, 3], boxes[:, None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
    area_i = (w * h)[:, :, None]
    area_j = (w * h)[:, None, :]
    union = (area_i + area_j - inter).clamp(min=1e-6)
    iou = (inter / union).unsqueeze(-1)
    center_dist = torch.sqrt((dxdy.square()).sum(dim=-1, keepdim=True).clamp(min=1e-9))
    area_ratio = torch.log(area_i / area_j.clamp(min=1e-6)).clamp(min=-5.0, max=5.0).unsqueeze(-1)
    aspect_i = (a[..., 2] / a[..., 3].clamp(min=1e-6)).unsqueeze(-1)
    aspect_j = (b[..., 2] / b[..., 3].clamp(min=1e-6)).unsqueeze(-1)
    aspect_ratio = torch.log(aspect_i / aspect_j.clamp(min=1e-6)).clamp(min=-5.0, max=5.0)
    return torch.cat([dxdy, log_wh, iou, center_dist, area_ratio, aspect_ratio], dim=-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for idx in range(max(1, num_layers)):
        in_dim = input_dim if idx == 0 else hidden_dim
        out_dim = output_dim if idx == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if idx < num_layers - 1:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _valid_num_heads(dim: int, requested: int) -> int:
    requested = max(1, min(int(requested), dim))
    for heads in range(requested, 0, -1):
        if dim % heads == 0:
            return heads
    return 1
