import argparse, torch, torch.nn as nn, torch.nn.functional as F, timm
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import DATASETS, YoloDataset, collate_fn, IMG_EXT
from src.model.vit_withConvEmbedding import ConvPatchEmbed


ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="foci")
ap.add_argument("--mode", default="distill", choices=["distill", "ae"])
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--img-size", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DS = DATASETS[args.dataset]; S = args.img_size or DS["img_size"]
out = args.out or f"pe_{args.mode}_{args.dataset}.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

def loader(split, aug):
    paths = sorted(p for p in (Path(DS["root"]) / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT)
    ds = YoloDataset(paths, S, augment=aug, mosaic=0.0, mixup=0.0)   # no mosaic: seams hurt reconstruction
    return DataLoader(ds, batch_size=args.bs, shuffle=aug, num_workers=4, collate_fn=collate_fn)

train, val = loader("train", True), loader("val", False)        # TRAIN images only; never test

enc = ConvPatchEmbed().to(dev)
if args.mode == "distill":
    teacher = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0, img_size=S)
    teacher = teacher.patch_embed.proj.to(dev).eval()           # original frozen 16x16 conv
    for p in teacher.parameters(): p.requires_grad = False
    dec, params = None, list(enc.parameters())
else:
    dec = nn.Sequential(
        nn.ConvTranspose2d(384, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.GroupNorm(8, 64),  nn.GELU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.GroupNorm(8, 32),  nn.GELU(),
        nn.ConvTranspose2d(32, 3, 4, 2, 1)).to(dev)
    params = list(enc.parameters()) + list(dec.parameters())

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

def step(x):
    z = enc(x)
    if args.mode == "distill":
        with torch.no_grad(): t = teacher(x)
        return F.mse_loss(z, t), z.std().item(), t.std().item()
    return F.mse_loss(dec(z), x), z.std().item(), x.std().item()

for ep in range(1, args.epochs + 1):
    enc.train(); tot = 0
    for x, _, _ in train:
        loss, zs, ts = step(x.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    if ep % 5 == 0 or ep == 1:
        enc.eval()
        with torch.no_grad():
            v = sum(step(x.to(dev))[0].item() for x, _, _ in val) / len(val)
        print(f"ep {ep:3d} | train {tot/len(train):.4f} | val {v:.4f} | tok std enc {zs:.3f} vs target {ts:.3f}")

torch.save(enc.state_dict(), out)       # encoder only; the decoder is discarded
print("saved", out)