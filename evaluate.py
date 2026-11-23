# evaluate.py
import argparse, csv, time
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from src.dataset import build_folder_loaders, DATASETS
from src.model.build import build_model, run_tag
from src.experiments import (
    ABLATIONS, InferenceOptions, add_inference_options, add_training_options, inference_options,
    validate_experiment_dataset,
)
from src.utils.prediction import detect_batch, detect_image, letterbox_geometry

USE_P2, BATCH_SIZE, IOU_THRES = False, 4, 0.65
device = "cuda" if torch.cuda.is_available() else "cpu"


def targets_to_xyxy(t, size):
    cx, cy, w, h = t[:, 1] * size, t[:, 2] * size, t[:, 3] * size, t[:, 4] * size
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), t[:, 0].long()


def _sync():
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_run(dataset, aug, seed, rates, split, kind, conf, img_size=None, *,
                 nwd=False, aux_center=False, options=InferenceOptions()):
    """One checkpoint -> dict of metrics, or None if the checkpoint doesn't exist."""
    DS = DATASETS[dataset]
    size = img_size or DS["img_size"]
    tag = run_tag(dataset, size, USE_P2, aug, seed, sac=rates, nwd=nwd, aux_center=aux_center)
    path = f"{kind}_{tag}.pt"
    if not Path(path).exists():
        print(f"  [skip] missing {path}")
        return None

    (loader,) = build_folder_loaders(DS["root"], splits=(split,), batch_size=BATCH_SIZE,
                                     img_size=size, augment=False, num_workers=2)
    model = build_model(
        size, len(DS["classes"]), use_p2=USE_P2, sac_rates=rates,
        aux_center=aux_center, pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    m25 = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.25])
    m.warn_on_many_detections = m25.warn_on_many_detections = False

    timings = []
    for images, targets, paths in loader:
        _sync()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            if options.sahi:
                dets = []
                for image_path in paths:
                    with Image.open(image_path) as source:
                        image = source.convert("RGB")
                    d = detect_image(model, image, size, device, conf, IOU_THRES, options)
                    # Keep all runs in the loader's evaluation coordinates, including small-object bins.
                    scale, pad_x, pad_y, _, _ = letterbox_geometry(image, size)
                    d[:, :4] = d[:, :4] * scale + d.new_tensor((pad_x, pad_y, pad_x, pad_y))
                    dets.append(d)
            else:
                dets = detect_batch(model, images.to(device), conf, IOU_THRES, options)
        _sync()
        timings.append((time.perf_counter() - t0, len(images)))
        preds, gts = [], []
        for d, t in zip(dets, targets):
            preds.append({"boxes": d[:, :4].cpu(), "scores": d[:, 4].cpu(),
                          "labels": d[:, 5].long().cpu()})
            b, l = targets_to_xyxy(t, size)
            gts.append({"boxes": b, "labels": l})
        m.update(preds, gts); m25.update(preds, gts)

    timed = timings[3:] or timings  # include warm-up when the validation split has at most 3 batches
    t_tot = sum(seconds for seconds, _ in timed)
    n_img = sum(count for _, count in timed)
    r, r25 = m.compute(), m25.compute()
    g = lambda x: float("nan") if x.item() < 0 else x.item()   # -1 = no GT of that size
    return dict(mAP25=r25["map"].item(), mAP50=r["map_50"].item(), mAP=r["map"].item(),
                mAP75=r["map_75"].item(), small=g(r["map_small"]), AR100=r["mar_100"].item(),
                params=params_m, ms=(t_tot / n_img * 1000) if n_img else float("nan"))


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
    head = "| Config | " + " | ".join(c for c, _ in cols) + " | Params (M) | ms/img | Δ mAP vs baseline (paired) |"
    sep = "|" + "---|" * (len(cols) + 4)
    lines = [f"**{split} / {kind} checkpoint** (mAP values ×100, mean ± std over seeds)", "", head, sep]
    for n in names:
        runs = res[n]
        if not runs:
            lines.append(f"| {n} (n=0) |" + " n/a |" * (len(cols) + 3)); continue
        row = [f"{n} (n={len(runs)})"]
        row += [ms_fmt([r[k] for r in runs.values()]) for _, k in cols]
        row += [ms_fmt([r["params"] for r in runs.values()], 2, 1.0),
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
    ap.add_argument("--configs", nargs="+", default=["none", "1,3,5"],
                    help='"none" = baseline (first one is the reference), e.g. none 1,3,5 1')
    ap.add_argument("--split", default=None, choices=["val", "test"])
    ap.add_argument("--ckpt", default="both", choices=["best", "last", "both"])
    ap.add_argument("--conf", type=float, default=0.001)
    add_training_options(ap)
    add_inference_options(ap)
    ap.add_argument("--ablations", nargs="+", choices=tuple(ABLATIONS), metavar="RUN",
                    help="compare named MRFP + QKV-SAC runs: e0 e1 e2 e3 e4 all")
    ap.add_argument("--sac-rates", default="1,3,5", help="SAC rates used by the named ablations")
    a = ap.parse_args()
    a.seeds = list(dict.fromkeys(a.seeds))
    options = inference_options(ap, a)
    validate_experiment_dataset(
        ap, a.dataset, a.ablations or a.nwd_loss or a.aux_center_loss or a.sahi or a.soft_nms,
    )
    if a.ablations and (a.nwd_loss or a.aux_center_loss or a.sahi or a.soft_nms):
        ap.error("use named --ablations or individual experiment flags")
    if not 0 <= a.conf <= 1:
        ap.error("--conf must be in [0, 1]")

    splits = DATASETS[a.dataset].get("splits", ("train", "val", "test"))
    split = a.split or ("test" if "test" in splits else "val")
    assert split in splits, f"{a.dataset} has no '{split}' split"

    if a.ablations:
        rates = tuple(int(x) for x in a.sac_rates.split(","))
        cfgs = {}
        for label in dict.fromkeys(["e0", *a.ablations]):
            nwd, center, sahi, soft_nms = ABLATIONS[label]
            cfgs[label.upper()] = (rates, nwd, center, replace(options, sahi=sahi, soft_nms=soft_nms))
    else:
        cfgs = {
            ("Baseline" if c == "none" else f"SAC d={c}"):
            (None if c == "none" else tuple(int(x) for x in c.split(",")),
             a.nwd_loss, a.aux_center_loss, options)
            for c in a.configs
        }
    names = list(cfgs)
    kinds = ["best", "last"] if a.ckpt == "both" else [a.ckpt]

    out_md, out_rows = [], []
    for kind in kinds:
        res = {n: {} for n in names}
        for n, (rates, nwd, center, post) in cfgs.items():
            for s in a.seeds:
                print(f"{kind} | {n} | seed {s}")
                r = evaluate_run(
                    a.dataset, a.aug, s, rates, split, kind, a.conf, a.img_size,
                    nwd=nwd, aux_center=center, options=post,
                )
                if r:
                    res[n][s] = r
                    out_rows.append(dict(
                        config=n, seed=s, ckpt=kind, split=split,
                        sac_rates=",".join(map(str, rates or ())), confidence=a.conf,
                        img_size=a.img_size or DATASETS[a.dataset]["img_size"],
                        nwd_loss=nwd, aux_center_loss=center, sahi=post.sahi,
                        soft_nms=post.soft_nms, tile_size=post.tile_size,
                        tile_overlap=post.tile_overlap, soft_nms_sigma=post.soft_nms_sigma, **r,
                    ))
        out_md.append(build_tables(res, names, split, kind))

    report = "\n\n".join(out_md)
    print("\n" + report)

    stem = f"results_{a.dataset}_aug{a.aug}_{split}"
    if a.ablations:
        stem += "_ablation_" + "-".join(names).lower()
    else:
        stem += "".join(suffix for enabled, suffix in (
            (a.nwd_loss, "_nwd"), (a.aux_center_loss, "_center"),
            (a.sahi, "_sahi"), (a.soft_nms, "_softnms"),
        ) if enabled)
    Path(f"{stem}.md").write_text(report)
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])) if out_rows else None
        if w:
            w.writeheader(); w.writerows(out_rows)
    print(f"\nsaved {stem}.md and {stem}.csv (per-seed rows)")
    if a.ablations:
        expected = len(cfgs) * len(a.seeds) * len(kinds)
        if len(out_rows) < expected:
            raise SystemExit("Ablation report is incomplete. See the missing checkpoint paths above.")
