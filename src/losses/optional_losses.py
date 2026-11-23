import torch
from torch import nn
from torch.nn import functional as F


class NWDBoxLoss(nn.Module):
    """Add NWD on the same foreground assignments and weights as the YOLO box loss."""

    def __init__(self, base, weight, scale):
        super().__init__()
        self.base = base
        self.weight = weight
        self.scale = scale

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, *extra, **kwargs):
        box, dfl = self.base(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                             target_scores, target_scores_sum, fg_mask, *extra, **kwargs)
        # In feature-grid units, W2 is the norm of (delta center, delta half-size).
        delta = pred_bboxes[fg_mask].float() - target_bboxes[fg_mask].float()
        left, right = delta.chunk(2, dim=-1)
        distance = torch.linalg.vector_norm(torch.cat((left + right, right - left), -1) * 0.5, dim=-1)
        penalty = -torch.expm1(-distance / self.scale)
        scores = target_scores[fg_mask].sum(-1)
        return box + self.weight * (penalty * scores).sum() / target_scores_sum, dfl


def center_loss(logits, batch):
    """Class-agnostic focal loss on occupied center cells of the finest pyramid map."""
    logits = logits.float().squeeze(1)
    _, height, width = logits.shape
    target = torch.zeros_like(logits)
    centers = batch["bboxes"][:, :2]
    x = (centers[:, 0] * width).long().clamp(0, width - 1)
    y = (centers[:, 1] * height).long().clamp(0, height - 1)
    target[batch["batch_idx"].long(), y, x] = 1
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    focal = bce * (-torch.expm1(-bce)).square()
    return focal.sum() / target.sum().clamp_min(1)
