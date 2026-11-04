#!/usr/bin/env python3
"""Colab-ready runner for the ViTDet-YOLO FOCI pipeline.

This script does the full workflow in one place:
  1) prepares the FOCI dataset using prepare_foci.py
  2) verifies the repo paths and model config
  3) trains the model on FOCI with train.py
  4) evaluates the saved checkpoints with evaluate.py

Usage in Google Colab:
    !git clone https://github.com/granth0ag/ViTDet-YOLO
    %cd ViTDet-YOLO
    !pip install -r requirements.txt
    !python vit_foci.py --epochs 100 --seeds 42 43 44

If the dataset is already prepared, it will reuse it automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_cmd(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n[RUN] {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    return completed


def ensure_kaggle_temp_link() -> None:
    """Keep /kaggle/temp in sync with the repo data directory when running on Kaggle."""
    kaggle_temp = Path("/kaggle/temp")
    if not Path("/kaggle").exists():
        return
    kaggle_temp.mkdir(parents=True, exist_ok=True)
    target = kaggle_temp / "foci"
    source = ROOT / "data" / "foci"
    if source.exists() and not target.exists():
        try:
            os.symlink(source, target)
            print(f"Created Kaggle symlink: {target} -> {source}")
        except OSError:
            shutil.copytree(source, target, dirs_exist_ok=True)


def ensure_dataset(val_size: int = 14, force: bool = False) -> None:
    """Run prepare_foci.py if the FOCI dataset has not been prepared yet."""
    dataset_dir = ROOT / "data" / "foci"
    train_dir = dataset_dir / "images" / "train"
    if train_dir.exists() and any(train_dir.iterdir()) and not force:
        print(f"FOCI dataset already prepared at: {dataset_dir}")
        ensure_kaggle_temp_link()
        return

    print("Preparing FOCI dataset...")
    run_cmd([PY, "prepare_foci.py", "--val-size", str(val_size)])
    ensure_kaggle_temp_link()


def verify_repo_imports() -> None:
    """Small sanity check that the project modules are importable."""
    code = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from src.dataset import DATASETS
from src.model.build import build_model
DS = DATASETS['foci']
model = build_model(DS['img_size'], len(DS['classes']), use_p2=False, sac_rates=None)
print(type(model).__name__, model.backbone.__class__.__name__)
"""
    run_cmd([PY, "-c", code], check=True)


def run_training(seeds: list[int], epochs: int, aug: str, sac_rates: str | None = None, close_mosaic: int = 10) -> None:
    """Train the model for the requested set of seeds."""
    for seed in seeds:
        cmd = [
            PY,
            "train.py",
            "--dataset",
            "foci",
            "--aug",
            aug,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--close-mosaic",
            str(close_mosaic),
        ]
        if sac_rates:
            cmd += ["--sac-rates", sac_rates]
        run_cmd(cmd, check=True)


def run_evaluation(seeds: list[int], aug: str, configs: list[str], split: str | None = None) -> None:
    """Evaluate the saved checkpoints for the requested configs."""
    cmd = [
        PY,
        "evaluate.py",
        "--dataset",
        "foci",
        "--aug",
        aug,
        "--seeds",
        *[str(s) for s in seeds],
        "--configs",
        *configs,
    ]
    if split:
        cmd += ["--split", split]
    run_cmd(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ViTDet-YOLO FOCI training pipeline in Colab.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation and reuse the existing data/foci folder.")
    parser.add_argument("--val-size", type=int, default=14, help="Validation split size for prepare_foci.py (default: 14).")
    parser.add_argument("--force-prepare", action="store_true", help="Re-run prepare_foci.py even if data/foci exists.")

    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for each seed.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to train with.")
    parser.add_argument("--aug", default="A", choices=["A", "B", "C", "D"], help="Augmentation preset from src/dataset.py.")
    parser.add_argument("--sac-rates", default="", help="Optional SAC rates, e.g. '1,3,5' or '1'.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epoch after which mosaic/mixup is disabled.")

    parser.add_argument("--skip-train", action="store_true", help="Skip training and only prepare/evaluate if checkpoints already exist.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--split", default=None, choices=["val", "test"], help="Evaluation split. Defaults to test if present else val.")
    parser.add_argument("--configs", nargs="+", default=["none", "1,3,5", "1"], help="Evaluate configs for evaluate.py, example: none 1,3,5 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Make sure the repo root is the working directory for all child scripts.
    os.chdir(ROOT)

    if not args.skip_prepare:
        ensure_dataset(val_size=args.val_size, force=args.force_prepare)

    print("\nChecking the model build path...")
    verify_repo_imports()

    if not args.skip_train:
        print("\nTraining the FOCI model(s)...")
        run_training(
            seeds=args.seeds,
            epochs=args.epochs,
            aug=args.aug,
            sac_rates=args.sac_rates or None,
            close_mosaic=args.close_mosaic,
        )

    if not args.skip_evaluate:
        print("\nEvaluating the trained checkpoints...")
        run_evaluation(
            seeds=args.seeds,
            aug=args.aug,
            configs=args.configs,
            split=args.split,
        )

    print("\nDone. The FOCI pipeline has finished.")


if __name__ == "__main__":
    main()
