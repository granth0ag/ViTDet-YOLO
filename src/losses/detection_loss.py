# detector_loss.py
import torch
from ultralytics.utils.loss import v8DetectionLoss
from src.losses.optional_losses import NWDBoxLoss, center_loss


def make_batch(targets, device):
   
    idx = torch.cat([torch.full((len(t),), i, dtype=torch.float32)
                     for i, t in enumerate(targets)])
    t = torch.cat(targets)
    return {
        "batch_idx": idx.to(device),       # [N]
        "cls":       t[:, :1].to(device),  # [N, 1]
        "bboxes":    t[:, 1:].to(device),  # [N, 4] cx, cy, w, h (0-1)
    }



class DetectionLoss:
    """Thin wrapper: model + targets list -> loss."""
    def __init__(self, model, nwd_weight=0.0, nwd_scale=1.6, aux_center_weight=0.0):
        self.criterion = v8DetectionLoss(model)   # model ko device pe bhejne ke BAAD banana

        self.aux_center_weight = aux_center_weight
        if bool(aux_center_weight) != (model.center_head is not None):
            raise ValueError("Enable both the auxiliary head and its loss weight together")
        if nwd_weight:
            self.criterion.bbox_loss = NWDBoxLoss(self.criterion.bbox_loss, nwd_weight, nwd_scale)

    def __call__(self, preds, targets, device):
        batch = make_batch(targets, device)
        if self.aux_center_weight:
            preds, centers = preds
        loss, loss_items = self.criterion(preds, batch)

        if isinstance(loss_items, dict):          # naye ultralytics versions
            vals = list(loss_items.values())
            loss_items = torch.stack([torch.as_tensor(v, dtype=torch.float32).detach().reshape(()).cpu()
                                      for v in vals])
        else:                                     # purane versions: tensor [box, cls, dfl]
            loss_items = loss_items.detach().cpu()

        total = loss.sum()
        if self.aux_center_weight:
            auxiliary = self.aux_center_weight * center_loss(centers, batch)
            total = total + auxiliary * centers.shape[0]  # YOLO sums over batch
            loss_items = torch.cat((loss_items, auxiliary.detach().cpu().reshape(1)))
        return total, loss_items
