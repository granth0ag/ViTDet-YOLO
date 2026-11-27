"""Run fixed experiment profiles without editing training code or preparing data."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from src.utils.run_config import run_name

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "cell-notebook": dict(dataset="foci_cell", lr="1e-4", workers="2",
                          variants=("none", "full"), seeds=(42,), conf="0.01", fp32=True),
    "foci-current": dict(dataset="foci", lr="3e-4", workers="4",
                         variants=("none", "plain-conv", "single-conv", "plain-conv-sac"),
                         seeds=(42, 43, 44), conf="0.001", fp32=False),
}


def commands(profile, seeds, epochs, name, eval_only=False):
    cfg = PROFILES[profile]
    shared = ["--dataset", cfg["dataset"], "--aug", "A", "--run-name", name]
    result = []
    if not eval_only:
        for variant in cfg["variants"]:
            for seed in seeds:
                result.append([sys.executable, str(ROOT / "train.py"), *shared,
                               "--epochs", str(epochs), "--seed", str(seed), "--mrfp", variant,
                               "--lr-backbone", cfg["lr"], "--num-workers", cfg["workers"]])
    result.append([sys.executable, str(ROOT / "evaluate.py"), *shared,
                   "--seeds", *map(str, seeds), "--configs", "none",
                   "--mrfp-variants", *cfg["variants"], "--split", "val", "--ckpt", "both",
                   "--conf", cfg["conf"], *(["--full-precision"] if cfg["fp32"] else [])])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, choices=PROFILES)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--run-name", type=run_name, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")
    cfg = PROFILES[args.profile]
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        ap.error("seeds must be unique integers between 0 and 2**32-1")
    name = args.run_name or f"{args.profile}-e{args.epochs}"
    planned = commands(args.profile, seeds, args.epochs, name, args.eval_only)
    print(f"Profile: {args.profile} | dataset={cfg['dataset']} | run-name={name}", flush=True)
    print(f"Variants: {', '.join(cfg['variants'])} | seeds={list(seeds)} | epochs={args.epochs}", flush=True)
    print("Plain ViT backbone; existing prepared data is reused without changes.", flush=True)
    for cmd in planned:
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return

    # Check the entire plan before starting an expensive run. No model downloads here.
    from src.dataset import DATASETS, IMG_EXT
    from src.utils.mrfp_config import mrfp_suffix
    from src.utils.run_config import run_suffix
    ds = DATASETS[cfg["dataset"]]
    data_root = ds["root"] if ds["root"].is_absolute() else ROOT / ds["root"]
    for split in ("train", "val"):
        image_dir, label_dir = data_root / "images" / split, data_root / "labels" / split
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMG_EXT)
        if not images or any(not (label_dir / (p.stem + ".txt")).is_file() for p in images):
            ap.error(f"Missing prepared images/labels in {data_root} ({split}). Run the existing dataset preparation first.")
    for variant in cfg["variants"]:
        for seed in seeds:
            tag = (f"{cfg['dataset']}_{ds['img_size']}_p3p5_augA_s{seed}"
                   + mrfp_suffix(variant) + run_suffix(name))
            for kind in ("best", "last"):
                checkpoint = ROOT / f"{kind}_{tag}.pt"
                if args.eval_only and not checkpoint.exists():
                    ap.error(f"Missing checkpoint: {checkpoint.name}")
                if not args.eval_only and checkpoint.exists():
                    ap.error(f"Existing checkpoint: {checkpoint.name}. Use --eval-only or a new --run-name.")
    for cmd in planned:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
