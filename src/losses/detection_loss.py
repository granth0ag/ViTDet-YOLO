# detector_loss.py
import torch
from ultralytics.utils.loss import v8DetectionLoss


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
    def __init__(self, model):
        self.criterion = v8DetectionLoss(model)   # model ko device pe bhejne ke BAAD banana

    def __call__(self, preds, targets, device):
        loss, loss_items = self.criterion(preds, make_batch(targets, device))

        if isinstance(loss_items, dict):          # naye ultralytics versions
            vals = list(loss_items.values())
            loss_items = torch.stack([torch.as_tensor(v, dtype=torch.float32).detach().reshape(()).cpu()
                                      for v in vals])
        else:                                     # purane versions: tensor [box, cls, dfl]
            loss_items = loss_items.detach().cpu()

        return loss.sum(), loss_items