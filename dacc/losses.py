from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .box_ops import generalized_box_iou


def ranker_loss(pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_scores, target_scores)


def dacc_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float] | None = None) -> tuple[torch.Tensor, Dict[str, float]]:
    weights = weights or {}
    pred_boxes = outputs["boxes"]
    pred_scores = outputs["scores"]
    action_logits = outputs["action_logits"]
    issue_logits = outputs["issue_logits"]

    tgt_boxes = batch["target_boxes"]
    tgt_scores = batch["target_scores"]
    tgt_actions = batch["target_actions"]
    tgt_issues = batch["target_issues"]
    mask = batch["target_mask"]

    # Query i supervises target i. This is intentionally simple for a stable baseline.
    q = min(pred_boxes.shape[1], tgt_boxes.shape[1])
    pred_boxes = pred_boxes[:, :q]
    pred_scores = pred_scores[:, :q]
    action_logits = action_logits[:, :q]
    issue_logits = issue_logits[:, :q]
    tgt_boxes = tgt_boxes[:, :q]
    tgt_scores = tgt_scores[:, :q]
    tgt_actions = tgt_actions[:, :q]
    tgt_issues = tgt_issues[:, :q]
    mask = mask[:, :q]

    if mask.any():
        l1 = F.l1_loss(pred_boxes[mask], tgt_boxes[mask])
        giou = 1.0 - torch.diag(generalized_box_iou(pred_boxes[mask], tgt_boxes[mask])).mean()
        score = F.smooth_l1_loss(pred_scores[mask], tgt_scores[mask])
        action = F.cross_entropy(action_logits[mask], tgt_actions[mask])
        issue = F.cross_entropy(issue_logits[mask], tgt_issues[mask])
    else:
        zero = pred_boxes.sum() * 0
        l1 = giou = score = action = issue = zero

    # Encourage query scores for padded targets to be low.
    if (~mask).any():
        neg_score = pred_scores[~mask].mean()
    else:
        neg_score = pred_scores.sum() * 0

    total = (
        weights.get("box_l1", 2.0) * l1
        + weights.get("giou", 2.0) * giou
        + weights.get("score", 0.5) * score
        + weights.get("action", 0.5) * action
        + weights.get("issue", 0.3) * issue
        + weights.get("neg_score", 0.1) * neg_score
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "loss_l1": float(l1.detach().cpu()),
        "loss_giou": float(giou.detach().cpu()),
        "loss_score": float(score.detach().cpu()),
        "loss_action": float(action.detach().cpu()),
        "loss_issue": float(issue.detach().cpu()),
        "loss_neg_score": float(neg_score.detach().cpu()),
    }
    return total, logs

