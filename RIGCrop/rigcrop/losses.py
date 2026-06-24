from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_crop_loss(winner_score: torch.Tensor, loser_score: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight / weight.mean().clamp(min=1e-6)
    return (F.softplus(-(winner_score - loser_score)) * weight).mean()


def graph_supervision_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    valid = batch["node_valid"].float()
    has_box = batch["node_has_box"].float() * valid
    denom_valid = valid.sum().clamp(min=1.0)
    denom_box = has_box.sum().clamp(min=1.0)

    bbox = F.smooth_l1_loss(out["node_boxes"], batch["node_boxes"], reduction="none").sum(dim=-1)
    loss_bbox = (bbox * has_box).sum() / denom_box

    role = F.cross_entropy(
        out["node_role_logits"].reshape(-1, out["node_role_logits"].size(-1)),
        batch["node_roles"].reshape(-1),
        reduction="none",
    ).view_as(valid)
    loss_role = (role * valid).sum() / denom_valid

    imp = F.smooth_l1_loss(out["node_importance"], batch["node_importance"], reduction="none")
    loss_importance = (imp * valid).sum() / denom_valid

    loss_valid = F.binary_cross_entropy_with_logits(out["node_valid_logits"], valid, reduction="mean")

    rel_mask = batch["relation_mask"].float()
    denom_rel = rel_mask.sum().clamp(min=1.0)
    rel_policy = F.cross_entropy(
        out["relation_logits"].reshape(-1, out["relation_logits"].size(-1)),
        batch["relation_policy"].reshape(-1),
        reduction="none",
    ).view_as(rel_mask)
    loss_relation_policy = (rel_policy * rel_mask).sum() / denom_rel
    rel_weight = F.smooth_l1_loss(out["relation_weight"], batch["relation_weight"], reduction="none")
    loss_relation_weight = (rel_weight * rel_mask).sum() / denom_rel

    loss_action = F.binary_cross_entropy_with_logits(out["action_logits"], batch["action_targets"], reduction="mean")

    return {
        "node_bbox": loss_bbox,
        "node_role": loss_role,
        "node_importance": loss_importance,
        "node_valid": loss_valid,
        "relation_policy": loss_relation_policy,
        "relation_weight": loss_relation_weight,
        "action": loss_action,
    }


def utility_distillation_loss(winner_utility: torch.Tensor, loser_utility: torch.Tensor, teacher_winner: torch.Tensor, teacher_loser: torch.Tensor) -> torch.Tensor:
    smooth = F.smooth_l1_loss(winner_utility, teacher_winner) + F.smooth_l1_loss(loser_utility, teacher_loser)
    teacher_margin = teacher_winner - teacher_loser
    mask = (teacher_margin.abs() > 0.05).float()
    if mask.sum() <= 0:
        return smooth
    sign = torch.where(teacher_margin >= 0, 1.0, -1.0)
    pair = F.softplus(-sign * (winner_utility - loser_utility))
    return smooth + (pair * mask).sum() / mask.sum().clamp(min=1.0)


def query_proposal_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "query_boxes" not in out or "query_scores" not in out:
        return out["score"].new_zeros(())
    target_boxes = batch["winner_box_feat"][:, :4]
    query_boxes = out["query_boxes"]
    query_scores = out["query_scores"]
    expanded_targets = target_boxes[:, None, :].expand_as(query_boxes)
    l1 = F.smooth_l1_loss(query_boxes, expanded_targets, reduction="none").sum(dim=-1)
    best_l1 = l1.min(dim=1).values.mean()
    iou_targets = _box_iou_with_target(query_boxes, target_boxes).detach()
    score_loss = F.binary_cross_entropy(query_scores.clamp(1e-4, 1.0 - 1e-4), iou_targets, reduction="mean")
    return best_l1 + score_loss


def _box_iou_with_target(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target[:, None, :]
    x1 = torch.maximum(boxes[..., 0], target[..., 0])
    y1 = torch.maximum(boxes[..., 1], target[..., 1])
    x2 = torch.minimum(boxes[..., 2], target[..., 2])
    y2 = torch.minimum(boxes[..., 3], target[..., 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    box_area = (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)
    target_area = (target[..., 2] - target[..., 0]).clamp(min=0.0) * (target[..., 3] - target[..., 1]).clamp(min=0.0)
    return inter / (box_area + target_area - inter).clamp(min=1e-6)
