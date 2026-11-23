import torch
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss


def make_batch(targets, device):
    idx = torch.cat([torch.full((len(t),), i, dtype=torch.float32)
                     for i, t in enumerate(targets)])
    t = torch.cat(targets)
    return {
        "batch_idx": idx.to(device),
        "cls": t[:, :1].to(device),
        "bboxes": t[:, 1:].to(device),
    }


class NWDBboxLoss(BboxLoss):
    """Add pixel-space normalized Wasserstein distance to the existing box term."""

    def __init__(self, reg_max, gain, scale):
        super().__init__(reg_max)
        self.gain, self.scale = gain, scale

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        ciou, dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride,
        )
        rows, anchors = fg_mask.nonzero(as_tuple=True)
        delta = (pred_bboxes[rows, anchors].float() - target_bboxes[rows, anchors].float())
        delta = delta * stride[anchors].float()
        # Gaussian embedding of an xyxy box: (cx, cy, width/2, height/2).
        top_left, bottom_right = delta.split(2, dim=-1)
        embedding_delta = torch.cat((top_left + bottom_right, bottom_right - top_left), dim=-1) / 2
        distance = torch.linalg.vector_norm(embedding_delta, dim=-1)
        nwd = -torch.expm1(-distance / self.scale)
        weight = target_scores[rows, anchors].float().sum(-1)
        return ciou + self.gain * (weight * nwd).sum() / target_scores_sum, dfl


class DetectionLoss:
    """YOLO loss with optional NWD and auxiliary center supervision."""

    def __init__(self, model, *, nwd=False, aux_center=False, nwd_gain=0.5,
                 nwd_scale=12.8, center_gain=0.25):
        self.criterion = v8DetectionLoss(model)
        self.aux_center = aux_center
        self.center_gain = center_gain
        if nwd:
            self.criterion.bbox_loss = NWDBboxLoss(
                self.criterion.reg_max, nwd_gain, nwd_scale,
            ).to(self.criterion.device)

    def __call__(self, preds, targets, device):
        batch = make_batch(targets, device)
        if self.aux_center:
            raw, offsets = preds
            assignment, parts, items = self.criterion.get_assigned_targets_and_loss(raw, batch)
            fg_mask, _, boxes, anchor_points, stride = assignment
            rows, anchors = fg_mask.nonzero(as_tuple=True)
            matched = boxes[rows, anchors].float()
            centers = (matched[:, :2] + matched[:, 2:]) / (2 * stride[anchors].float())
            target_offsets = centers - anchor_points[anchors].float()
            center = F.smooth_l1_loss(
                offsets[rows, anchors].float(), target_offsets, reduction="sum",
            ) / max(1, 2 * rows.numel())
            loss = (parts.sum() + self.center_gain * center) * offsets.shape[0]
            items = {**items, "center": (self.center_gain * center).detach()}
        else:
            loss, items = self.criterion(preds, batch)

        if isinstance(items, dict):
            items = torch.stack(list(items.values()))
        return loss.sum(), items.detach().cpu()
