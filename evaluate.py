# evaluate.py
import argparse, csv, time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from src.dataset import build_folder_loaders, DATASETS
from src.model.build import build_model, run_tag
from src.utils.mrfp_config import (
    add_mrfp_args, requested_variants, resolve_mrfp, model_label, mrfp_report_suffix,
)

USE_P2, BATCH_SIZE, IOU_THRES = False, 4, 0.65
device = "cuda" if torch.cuda.is_available() else "cpu"


def targets_to_xyxy(t, size):
    cx, cy, w, h = t[:, 1] * size, t[:, 2] * size, t[:, 3] * size, t[:, 4] * size
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), t[:, 0].long()


def _sync():
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_run(dataset, aug, seed, rates, split, kind, conf, img_size=None, mrfp_variant=None):
    """One checkpoint -> dict of metrics, or None if the checkpoint doesn't exist."""
    DS = DATASETS[dataset]
    size = img_size or DS["img_size"]
    path = f"{kind}_{run_tag(dataset, size, USE_P2, aug, seed, sac=rates, mrfp_variant=mrfp_variant)}.pt"
    if not Path(path).exists():
        print(f"  [skip] missing {path}")
        return None

    (loader,) = build_folder_loaders(DS["root"], splits=(split,), batch_size=BATCH_SIZE,
                                     img_size=size, augment=False, num_workers=2)
    model = build_model(size, len(DS["classes"]), use_p2=USE_P2, sac_rates=rates,
                        mrfp_variant=mrfp_variant).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    mrfp = getattr(model.neck, "mrfp", None)
    mrfp_params_m = sum(p.numel() for p in mrfp.parameters()) / 1e6 if mrfp is not None else 0.0

    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    m25 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.25])
    m.warn_on_many_detections = m25.warn_on_many_detections = False

    t_tot, n_img = 0.0, 0
    for bi, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        _sync(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            y, _ = model(images)
        _sync(); dt = time.perf_counter() - t0
        if bi >= 3:                                   # skip warm-up batches
            t_tot += dt; n_img += len(images)

        dets = non_max_suppression(y, conf_thres=conf, iou_thres=IOU_THRES)
        preds, gts = [], []
        for d, t in zip(dets, targets):
            preds.append({"boxes": d[:, :4].cpu(), "scores": d[:, 4].cpu(),
                          "labels": d[:, 5].long().cpu()})
            b, l = targets_to_xyxy(t, size)
            gts.append({"boxes": b, "labels": l})
        m.update(preds, gts); m25.update(preds, gts)

    r, r25 = m.compute(), m25.compute()
    g = lambda x: float("nan") if x.item() < 0 else x.item()   # -1 = no GT of that size
    return dict(mAP25=r25["map"].item(), mAP50=r["map_50"].item(), mAP=r["map"].item(),
                mAP75=r["map_75"].item(), small=g(r["map_small"]), AR100=r["mar_100"].item(),
                params=params_m, mrfp_params=mrfp_params_m,
                ms=(t_tot / n_img * 1000) if n_img else float("nan"))


# ---------------- table helpers ----------------
def ms_fmt(vals, d=2, scale=100.0):
    v = np.array(vals, float) * scale
    v = v[~np.isnan(v)]
    if not len(v):
        return "n/a"
    return f"{v.mean():.{d}f} ± {(v.std(ddof=1) if len(v) > 1 else 0.0):.{d}f}"


def paired_delta(base, other, key="mAP"):
    diffs = [(other[s][key] - base[s][key]) * 100 for s in other if s in base]
    if not diffs:
        return "n/a"
    wins = sum(d > 0 for d in diffs)
    return f"{ms_fmt(diffs, scale=1.0)} ({wins}/{len(diffs)} seeds +)"


def build_tables(res, names, split, kind):
    cols = [("mAP@.25", "mAP25"), ("mAP@.5", "mAP50"), ("mAP", "mAP"), ("mAP@.75", "mAP75"),
            ("mAP small", "small"), ("AR@100", "AR100")]
    base = res[names[0]]
    head = "| Config | " + " | ".join(c for c, _ in cols) + " | Params (M) | MRFP params (M) | ms/img | Δ mAP vs first config (paired) |"
    sep = "|" + "---|" * (len(cols) + 5)
    lines = [f"**{split} / {kind} checkpoint** (mAP values ×100, mean ± std over seeds)", "", head, sep]
    for n in names:
        runs = res[n]
        if not runs:
            lines.append(f"| {n} (n=0) |" + " n/a |" * (len(cols) + 4)); continue
        row = [f"{n} (n={len(runs)})"]
        row += [ms_fmt([r[k] for r in runs.values()]) for _, k in cols]
        row += [ms_fmt([r["params"] for r in runs.values()], 2, 1.0),
                ms_fmt([r["mrfp_params"] for r in runs.values()], 4, 1.0),
                ms_fmt([r["ms"] for r in runs.values()], 1, 1.0),
                "—" if n == names[0] else paired_delta(base, runs)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------- main ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="foci", choices=list(DATASETS))
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--aug", default="A", choices=["A", "B", "C", "D"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--configs", nargs="+", default=None,
                    help='SAC backbone configs: "none" = plain ViT; defaults to none for explicit MRFP '
                         'experiments, otherwise the legacy none 1,3,5 comparison')
    ap.add_argument("--split", default=None, choices=["val", "test"])
    ap.add_argument("--ckpt", default="both", choices=["best", "last", "both"])
    ap.add_argument("--conf", type=float, default=0.001)
    add_mrfp_args(ap, sweep=True)
    a = ap.parse_args()
    variants = requested_variants(a)
    configs = a.configs if a.configs is not None else (["none"] if variants != [None] else ["none", "1,3,5"])

    splits = DATASETS[a.dataset].get("splits", ("train", "val", "test"))
    split = a.split or ("test" if "test" in splits else "val")
    assert split in splits, f"{a.dataset} has no '{split}' split"

    cfgs = {}
    for c in configs:
        rates = None if c == "none" else tuple(int(x) for x in c.split(","))
        for variant in variants:
            cfgs[model_label(rates, variant)] = (rates, variant)
    names = list(cfgs)
    kinds = ["best", "last"] if a.ckpt == "both" else [a.ckpt]

    out_md, out_rows = [], []
    for kind in kinds:
        res = {n: {} for n in names}
        for n, (rates, variant) in cfgs.items():
            for s in a.seeds:
                print(f"{kind} | {n} | seed {s}")
                r = evaluate_run(a.dataset, a.aug, s, rates, split, kind, a.conf, a.img_size,
                                  mrfp_variant=variant)
                if r:
                    res[n][s] = r
                    out_rows.append(dict(config=n, seed=s, ckpt=kind, split=split,
                                         sac_rates="none" if rates is None else ",".join(map(str, rates)),
                                         mrfp_variant=resolve_mrfp(variant, rates), **r))
        out_md.append(build_tables(res, names, split, kind))

    report = "\n\n".join(out_md)
    print("\n" + report)

    stem = f"results_{a.dataset}_aug{a.aug}_{split}" + mrfp_report_suffix(variants)
    if variants != [None]:
        stem += "_bb-" + "_".join("vit" if c == "none" else "sac" + c.replace(",", "-") for c in configs)
    Path(f"{stem}.md").write_text(report)
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])) if out_rows else None
        if w:
            w.writeheader(); w.writerows(out_rows)
    print(f"\nsaved {stem}.md and {stem}.csv (per-seed rows)")
