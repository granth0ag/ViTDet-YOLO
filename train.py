# train.py
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import build_coco_subset_loaders
from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel
from src.losses.detection_loss import DetectionLoss
from src.utils.tb_utils import evaluate_map, detection_images

# ---------------- config ----------------
IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 100         # pehle test ke liye 2 kar ke chala
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
SEED = 42
EVAL_EVERY = 5            # test mAP + detection images kitne epochs baad (sirf monitoring)
NUM_WORKERS = 4
DATA_ROOT = "/kaggle/temp/coco_subset" if Path("/kaggle").exists() else "data/coco_subset"
LOG_DIR = "runs/coco3k_run1"
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- data ----------------
train_loader, test_loader = build_coco_subset_loaders(
    root=DATA_ROOT, batch_size=BATCH_SIZE, img_size=IMG_SIZE,
    augment=True, num_workers=NUM_WORKERS)
print(f"train: {len(train_loader.dataset)} images | test: {len(test_loader.dataset)} images")

# ---------------- model ----------------
backbone = ViTDetBackbone(img_size=IMG_SIZE)
neck = SimpleFeaturePyramid(in_ch=384, out_ch=128)
head = YoloHead(num_classes=80, ch=(128, 128, 128))
model = DetectionModel(backbone, neck, head).to(device)
criterion = DetectionLoss(model)

optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model.neck.parameters(),     "lr": LR_HEAD},
    {"params": model.head.parameters(),     "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

writer = SummaryWriter(LOG_DIR, flush_secs=30)


def train_one_epoch():
    model.train()
    total, items = 0.0, torch.zeros(3)
    for images, targets, _ in train_loader:
        images = images.to(device)
        preds = model(images)
        loss, loss_items = criterion(preds, targets, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total += loss.item()
        items += loss_items[:3].cpu()
    n = len(train_loader)
    return total / n, items / n


for epoch in range(1, EPOCHS + 1):
    tr_loss, it = train_one_epoch()

    # ---- loss + lr curves ----
    writer.add_scalar("loss/total", tr_loss, epoch)
    writer.add_scalar("loss/box", it[0], epoch)
    writer.add_scalar("loss/cls", it[1], epoch)
    writer.add_scalar("loss/dfl", it[2], epoch)
    writer.add_scalar("lr/backbone", optimizer.param_groups[0]["lr"], epoch)
    writer.add_scalar("lr/head", optimizer.param_groups[1]["lr"], epoch)
    scheduler.step()

    print(f"epoch {epoch:3d} | loss {tr_loss:8.3f} (box {it[0]:.3f} cls {it[1]:.3f} dfl {it[2]:.3f})")

    # ---- test mAP curves + detection images (monitoring only, isse kuch chunna nahi) ----
    if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
        r = evaluate_map(model, test_loader, device, IMG_SIZE)
        for k in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]:
            writer.add_scalar(f"metrics/{k}", r[k].item(), epoch)
        writer.add_images("detections (green=GT, red=pred)",
                          detection_images(model, test_loader, device, n=4, img_size=IMG_SIZE), epoch)
        print(f"          mAP@0.5 {r['map_50'].item():.4f} | mAP {r['map'].item():.4f}")
        torch.save(model.state_dict(), "last.pt")      # session toote to bhi bacha rahe

torch.save(model.state_dict(), "last.pt")
writer.close()