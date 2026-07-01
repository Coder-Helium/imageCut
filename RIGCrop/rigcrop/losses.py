from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def pairwise_crop_loss(winner_score: torch.Tensor, loser_score: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight / weight.mean().clamp(min=1e-6)
    return (F.softplus(-(winner_score - loser_score)) * weight).mean()


def listwise_crop_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.35,
) -> torch.Tensor:
    """ListNet-style loss for GAICD candidate MOS ranking.

    Args:
        pred_scores: B x C predicted crop logits.
        target_scores: B x C human MOS/final scores.
        mask: B x C valid candidate mask.
        temperature: lower values focus more probability mass on top-MOS crops.
    """
    mask = mask.bool()
    temp = max(float(temperature), 1e-6)
    pred = pred_scores.masked_fill(~mask, -1e4)
    target = (target_scores / temp).masked_fill(~mask, -1e4)
    target_prob = torch.softmax(target, dim=-1).detach()
    log_prob = torch.log_softmax(pred, dim=-1)
    per_item = -(target_prob * log_prob).sum(dim=-1)
    valid_rows = mask.sum(dim=-1) > 1
    if not valid_rows.any():
        return pred_scores.new_zeros(())
    return per_item[valid_rows].mean()


def topk_hard_negative_loss(
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    mask: torch.Tensor,
    positive_topk: int = 5,
    negative_after: int = 10,
    margin: float = 1.0,
) -> torch.Tensor:
    """Push GAICD top-MOS crops above high-scoring hard negatives."""
    losses = []
    pos_k = max(1, int(positive_topk))
    neg_after = max(pos_k + 1, int(negative_after))
    for pred, target, valid in zip(pred_scores, target_scores, mask.bool()):
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        if idx.numel() <= pos_k:
            continue
        order = idx[torch.argsort(target[idx], descending=True)]
        pos_idx = order[: min(pos_k, order.numel())]
        neg_idx = order[min(neg_after, order.numel()) :]
        if neg_idx.numel() == 0:
            neg_idx = order[pos_idx.numel() :]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        hard_pos = pred[pos_idx].min()
        hard_neg = pred[neg_idx].max()
        losses.append(F.softplus(hard_neg - hard_pos + float(margin)))
    if not losses:
        return pred_scores.new_zeros(())
    return torch.stack(losses).mean()


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
