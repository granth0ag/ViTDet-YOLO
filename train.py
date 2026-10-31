# train.py
import argparse
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import build_folder_loaders, DATASETS,AUG_PRESETS
from src.model.build import build_model, run_tag
from src.losses.detection_loss import DetectionLoss
from src.utils.tb_utils import evaluate_map, detection_images



# ---------------- args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--aug", default="D", choices=list(AUG_PRESETS), help="augmentation preset")

ap.add_argument("--patch-embed", default="linear", choices=["linear", "conv"])
ap.add_argument("--pe-ckpt", default=None, help="pretrained patch-embed weights (conv only)")
ap.add_argument("--lr-pe", type=float, default=1e-3, help="LR for the conv patch embed")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--backbone", default="vit", choices=["vit", "conv"], help="ViT backbone type")

ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--close-mosaic", type=int, default=10, help="aakhri itne epochs mein mosaic + mixup band")
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--dataset", default="wildlife", choices=list(DATASETS))
ap.add_argument("--img-size", type=int, default=None, help="override the dataset's default")
args = ap.parse_args()

# ---------------- config ----------------

BATCH_SIZE = 4
EPOCHS = args.epochs
CLOSE_MOSAIC = args.close_mosaic
EVAL_EVERY = args.eval_every
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
DROP_PATH = 0.1
USE_P2 = False            # True = stride-4 level (evaluate.py / inference.py mein SAME hona chahiye)
SEED = args.seed
NUM_WORKERS = 4
DS = DATASETS[args.dataset]
IMG_SIZE = args.img_size or DS["img_size"]
CLASS_NAMES, DATA_ROOT = DS["classes"], DS["root"]
NUM_CLASSES = len(CLASS_NAMES)
TAG = run_tag(args.dataset, IMG_SIZE, USE_P2, args.patch_embed, args.pe_ckpt, args.aug, SEED)
RUN = args.aug
AUG = AUG_PRESETS[RUN]
LOG_DIR = f"runs/{TAG}"
BEST_CKPT, LAST_CKPT = f"best_{TAG}.pt", f"last_{TAG}.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SPLITS = DS.get("splits", ("train", "val", "test"))
# ---------------- data ----------------
train_loader, val_loader = build_folder_loaders(
    DATA_ROOT, splits=SPLITS, batch_size=BATCH_SIZE,
    img_size=IMG_SIZE, augment=True, num_workers=NUM_WORKERS, aug=AUG)
print(f"run: {TAG} | backbone={args.backbone} | aug {RUN}: {AUG}")
print(f"epochs={EPOCHS} close_mosaic={CLOSE_MOSAIC} eval_every={EVAL_EVERY}")
print(f"train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} images")

# ---------------- model ----------------
model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2, drop_path_rate=DROP_PATH,
                    patch_embed=args.patch_embed, pe_ckpt=args.pe_ckpt).to(device)
criterion = DetectionLoss(model)                 # model .to(device) ke BAAD




pe_params = list(model.backbone.vit.patch_embed.parameters())
pe_ids = {id(p) for p in pe_params}
rest = [p for p in model.backbone.parameters() if id(p) not in pe_ids]
LR_PE = args.lr_pe if args.patch_embed == "conv" else LR_BACKBONE

optimizer = torch.optim.AdamW([
    {"params": rest,                    "lr": LR_BACKBONE},   # groups 0,1,2 keep their meaning
    {"params": model.neck.parameters(), "lr": LR_HEAD},       # for your TensorBoard logging
    {"params": model.head.parameters(), "lr": LR_HEAD},
    {"params": pe_params,               "lr": LR_PE},
], weight_decay=WEIGHT_DECAY)
assert sum(p.numel() for g in optimizer.param_groups for p in g["params"]) == \
       sum(p.numel() for p in model.parameters())           # no parameter left out

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

writer = SummaryWriter(LOG_DIR, flush_secs=30)





def train_one_epoch():
    model.train()
    total, items = 0.0, torch.zeros(3)

    for images, targets, _ in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            preds = model(images)
            loss, loss_items = criterion(preds, targets, device)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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