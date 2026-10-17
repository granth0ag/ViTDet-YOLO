# train.py
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import build_folder_loaders, NUM_CLASSES, DATA_ROOT
from src.model.build import build_model
from src.losses.detection_loss import DetectionLoss
from src.utils.tb_utils import evaluate_map, detection_images

# ---------------- config ----------------
IMG_SIZE = 640
BATCH_SIZE = 4           
EPOCHS = 3               
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
DROP_PATH = 0.1
USE_P2 = False            # True = stride-4 level (evaluate.py / inference.py mein SAME hoga)
SEED = 42
EVAL_EVERY = 1            # val mAP + detection images kitne epochs baad
NUM_WORKERS = 4

# ---------------- augmentation ablation ----------------
RUN = "D"                 # "A" / "B" / "C" / "D"
AUG_PRESETS = {
    "A": dict(mosaic=0.0, mixup=0.0, erase=0.0,  photo_ops=0),   # baseline: zoom/shift + colour jitter + flip
    "B": dict(mosaic=1.0, mixup=0.0, erase=0.0,  photo_ops=0),   # + mosaic
    "C": dict(mosaic=1.0, mixup=0.0, erase=0.25, photo_ops=2),   # + RandAugment-lite + random erasing
    "D": dict(mosaic=1.0, mixup=0.1, erase=0.25, photo_ops=2),   # + mixup
}
AUG = AUG_PRESETS[RUN]
CLOSE_MOSAIC = 1         # aakhri itne epochs mein mosaic + mixup band (clean images pe fine-tune)

TAG = f"wildlife_{IMG_SIZE}_{'p2' if USE_P2 else 'p3p5'}_aug{RUN}"
LOG_DIR = f"runs/{TAG}"
BEST_CKPT, LAST_CKPT = f"best_{TAG}.pt", f"last_{TAG}.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- data ----------------
train_loader, val_loader, _ = build_folder_loaders(
    DATA_ROOT, splits=("train", "val", "test"), batch_size=BATCH_SIZE,
    img_size=IMG_SIZE, augment=True, num_workers=NUM_WORKERS, aug=AUG)
print(f"run {RUN}: {AUG}")
print(f"train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} images")

# ---------------- model ----------------
model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2, drop_path_rate=DROP_PATH).to(device)
criterion = DetectionLoss(model)                 # model .to(device) ke BAAD

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


best_map = -1.0
for epoch in range(1, EPOCHS + 1):
    # ---- close mosaic: aakhri epochs clean images pe ----
    if epoch == EPOCHS - CLOSE_MOSAIC + 1 and (AUG["mosaic"] > 0 or AUG["mixup"] > 0):
        train_loader.dataset.mosaic_p = 0.0      # persistent_workers=False hai, to workers ko naya value mil jaata hai
        train_loader.dataset.mixup_p = 0.0
        print(f"epoch {epoch}: mosaic + mixup off")

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

    # ---- val mAP curves + detection images + best checkpoint ----
    if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
        r = evaluate_map(model, val_loader, device, IMG_SIZE, conf=0.01)
        for k in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]:
            writer.add_scalar(f"metrics/{k}", r[k].item(), epoch)
        writer.add_images("detections (green=GT, red=pred)",
                          detection_images(model, val_loader, device, n=4, img_size=IMG_SIZE), epoch)
        print(f"          val mAP@0.5 {r['map_50'].item():.4f} | mAP {r['map'].item():.4f}")

        if r["map"].item() > best_map:
            best_map = r["map"].item()
            torch.save(model.state_dict(), BEST_CKPT)
        torch.save(model.state_dict(), LAST_CKPT)      # session toote to bhi bacha rahe

torch.save(model.state_dict(), LAST_CKPT)
writer.close()
print(f"best val mAP: {best_map:.4f} -> {BEST_CKPT}")