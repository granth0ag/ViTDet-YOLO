# smoke.py
import time
import torch

from src.dataset import build_folder_loaders, NUM_CLASSES, DATA_ROOT
from src.model.build import build_model
from src.losses.detection_loss import DetectionLoss
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

# ---- train.py jaisa EXACT same rakh ----
IMG_SIZE = 640
USE_P2 = False
BATCH_SIZE = 4
DROP_PATH = 0.1
EPOCHS = 60

device = "cuda" if torch.cuda.is_available() else "cpu"
print("1) device:", device, "| torch", torch.__version__)

# ---- data (augment=True, taaki augmentation path bhi test ho) ----
(train_loader,) = build_folder_loaders(DATA_ROOT, splits=("train",), batch_size=BATCH_SIZE,
                                       img_size=IMG_SIZE, augment=True, num_workers=2)
images, targets, paths = next(iter(train_loader))
print("2) data:", tuple(images.shape), "| targets:", len(targets), tuple(targets[0].shape),
      "| train steps/epoch:", len(train_loader))
ids = torch.cat([t[:, 0] for t in targets])
assert ids.min() >= 0 and ids.max() < NUM_CLASSES, f"class id range galat: {ids.min()}..{ids.max()}"
for t in targets:                                        # boxes normalized aur valid hone chahiye
    if len(t):
        assert (t[:, 1:] >= 0).all() and (t[:, 1:] <= 1).all(), "box coords [0,1] se bahar"
        assert (t[:, 3:] > 0).all(), "zero/negative w ya h"

# ---- model ----
model = build_model(IMG_SIZE, NUM_CLASSES, use_p2=USE_P2, drop_path_rate=DROP_PATH).to(device)
criterion = DetectionLoss(model)
images = images.to(device)
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

# ---- forward shapes ----
model.train()
feat = model.backbone(images)
print("3) backbone:", tuple(feat.shape))                # [B, 384, 40, 40]
print("   neck    :", [tuple(p.shape) for p in model.neck(feat)])   # 80x80, 40x40, 20x20 (+160x160 agar P2)
preds = model(images)
if isinstance(preds, dict):                             # naye ultralytics
    print("   head keys:", list(preds.keys()))
    for k, v in preds.items():
        if torch.is_tensor(v):
            print(f"     {k}: {tuple(v.shape)}")
        else:
            print(f"     {k}:", [tuple(p.shape) for p in v])
else:                                                   # purane versions
    print("   head    :", [tuple(p.shape) for p in preds])

# ---- loss + gradients ----
loss, items = criterion(preds, targets, device)
print("4) loss:", loss.item(), "| items:", items)
assert torch.isfinite(loss), "loss NaN/inf"
loss.backward()
for name, m in [("backbone", model.backbone), ("neck", model.neck), ("head", model.head)]:
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    print(f"   grad in {name}: {has_grad}")

# ---- overfit ONE batch ----
opt = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.neck.parameters(),     "lr": 1e-3},
    {"params": model.head.parameters(),     "lr": 1e-3},
], weight_decay=0.05)

print("5) overfit 1 batch, 40 steps")
if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for step in range(40):
    preds = model(images)
    loss, items = criterion(preds, targets, device)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    if step % 10 == 0 or step == 39:
        print(f"   step {step:2d} loss {loss.item():8.3f}")
if device == "cuda":
    torch.cuda.synchronize()
dt = (time.time() - t0) / 40
epoch_s = dt * len(train_loader)
print(f"   {dt:.3f} s/step -> ~{epoch_s:.0f} s/epoch -> ~{epoch_s * EPOCHS / 3600:.1f} h for {EPOCHS} epochs"
      " (dataloading + eval ke bina, to asli time thoda zyada)")
if device == "cuda":
    print(f"   peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# ---- eval path ----
model.eval()
with torch.no_grad():
    y, _ = model(images)
print("6) eval y:", tuple(y.shape))                     # [B, 4+4, 8400]  (P2 ke saath 34000)
dets = non_max_suppression(y, conf_thres=0.001, iou_thres=0.65)
print("   NMS ok, dets per image:", [len(d) for d in dets])

# ---- checkpoint roundtrip ----
torch.save(model.state_dict(), "smoke.pt")
model.load_state_dict(torch.load("smoke.pt", map_location=device))
print("7) save/load ok\nSMOKE TEST PASSED")