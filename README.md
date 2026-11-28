# ViTDet-YOLO for FOCI and cell-patch detection

This PyTorch detector draws bounding boxes around **virus foci** and **small cell patches** in microscopy images. These are two separate, single-class detection tasks. The project studies whether giving a Vision Transformer access to different local neighborhoods improves box localization.

On the virus-patch validation split, adding switchable atrous convolution (SAC) to the backbone and multi-receptive-field refinement (MRFP) to the neck raised mean AP@0.50 from **78.84% to 80.47%** across three matched training seeds. Cell-patch localization was harder. In the earlier report, AP@0.75 ranged from **3.21% to 4.34%** across four cell models. A separate seed-42 neck-only experiment now reaches **47.59% AP@0.50 and 5.19% AP@0.75**, using the original dense residual MRFP and a plain ViT backbone on five validation images.

## Detector

A pretrained ViT-S/16 encodes a 640 × 640 image into a 40 × 40 feature map. A Simple Feature Pyramid builds three resolutions from that map, and a YOLOv8 head predicts boxes and scores at each resolution.

The combined detector in the report adds two operations. Before the ViT's query, key and value projections, **depthwise** SAC branches use dilation rates 1, 3 and 5. After the pyramid, a shared MRFP neck uses a learned spatial gate to mix two **dense** 3 × 3 convolutions with dilations 1 and 3. The reported improvements measure the two changes together. [Architecture and implementation details](docs/ARCHITECTURE.md)

## Virus-patch detection

The report describes 149 images, with 134 for training and 15 for validation. The following scores are validation percentages averaged over seeds 42, 43 and 44 after 200-epoch runs.

| Metric | Plain ViT | Backbone SAC + dense MRFP neck | Change (percentage points) |
| --- | ---: | ---: | ---: |
| mAP@0.25 | 87.12 | 87.87 | +0.75 |
| mAP@0.50 | 78.84 | 80.47 | +1.63 |
| mAP@0.75 | 16.17 | 17.95 | +1.78 |
| mAP@0.50:0.95 | 31.84 | 32.67 | +0.83 |
| mAP small | 30.80 | 29.97 | −0.83 |
| AR@100 | 44.67 | 44.90 | +0.23 |

AP@0.50 improved in all three seed pairs. The higher overall AP came with lower mean AP on small objects. [Per-seed scores and full metrics](docs/EXPERIMENTS.md#virus-patch-detection)

## Cell-patch detection

The cell-patch experiments used a **200-epoch training schedule**, with 45 training images and five validation images. All annotated validation objects fall in the small-object category. Table 8 reports the following validation scores (%).

| Metric | Plain ViT | SAC in the ViT backbone | SAC in the YOLOv8 head | Backbone SAC + dense MRFP neck |
| --- | ---: | ---: | ---: | ---: |
| mAP@0.25 | 53.61 | 53.66 | 52.25 | 54.89 |
| mAP@0.50 | 45.57 | 43.74 | 42.28 | 46.42 |
| mAP@0.75 | 3.94 | 4.34 | 3.21 | 3.85 |
| mAP@0.50:0.95 | 15.33 | 15.28 | 13.34 | 15.48 |
| AR@100 | 20.50 | 20.43 | 18.51 | 20.60 |

The first three model columns use final-epoch evaluations. The combined model column uses the latest pre-final validation snapshot reported in Table 8. It reached 46.42% mAP@0.50 and 15.48% mAP@0.50:0.95. Backbone SAC had the highest mAP@0.75 (4.34%). [Checkpoint details](docs/EXPERIMENTS.md#cell-patch-detection)

## New cell result: original residual MRFP, plain ViT backbone

A separate **seed-42, 200-epoch** run compares `--mrfp none` against `--mrfp full` on the same 45/5 cell split. The ViT backbone stays plain; SAC-style switching is inside the original dense residual MRFP neck. This is not the channel-split CoMer-style `plain-conv-sac` variant or the earlier combined backbone-SAC model.

**Best validation AP checkpoints**

| Metric | Plain ViT | Original dense residual MRFP | Change (percentage points) |
| --- | ---: | ---: | ---: |
| mAP@0.25 | 53.51 | 53.75 | +0.24 |
| mAP@0.50 | 45.30 | 47.59 | +2.29 |
| mAP@0.75 | 4.27 | 5.19 | +0.92 |
| mAP@0.50:0.95 | 15.94 | 15.99 | +0.04† |
| mAP small | 15.94 | 15.99 | ≈+0.05 |
| AR@100 | 20.95 | 20.72 | −0.23 |

**Last checkpoints — epoch 200**

| Metric | Plain ViT | Original dense residual MRFP | Change (percentage points) |
| --- | ---: | ---: | ---: |
| mAP@0.25 | 53.45 | 53.75 | +0.30 |
| mAP@0.50 | 45.29 | 47.59 | +2.30 |
| mAP@0.75 | 4.06 | 5.19 | +1.13 |
| mAP@0.50:0.95 | 15.64 | 15.99 | +0.34† |
| mAP small | 15.64 | 15.99 | ≈+0.35 |
| AR@100 | 20.68 | 20.72 | +0.04 |

† COCO AP deltas are copied from the evaluator's unrounded paired calculation. Other changes are calculated from displayed rounded scores.

Scores are validation percentages. At epoch 200, AP@0.50 improves by **2.30 percentage points**. The evaluator reports paired COCO AP gains of **+0.04 pp (best)** and **+0.34 pp (last)** using unrounded values; subtracting the displayed rounded scores can differ by 0.01 pp. Best-checkpoint AR@100 decreases by 0.23 pp. This is one seed, not a three-seed or statistical-significance result.

[Verified Kaggle run, version 42](https://www.kaggle.com/code/granthag/vit-foci?scriptVersionId=374856046) · [Full settings and provenance](docs/CELL_NOTEBOOK_RESULTS.md) · [Rounded results CSV](results/cell_notebook_seed42_kaggle_v42.csv)

To run the same profile on already prepared cell data:

```bash
python run_isolated.py --profile cell-notebook
```

## Run a microscopy experiment

Install the project dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

| Task | Dataset flag | Prepare the data |
|---|---|---|
| Virus foci | `--dataset foci` | Place the source folders `JPEGImages/JPEGImages/` and `Virus_anno/Annotations/` under `data/foci_raw/`, then run `python prepare_foci.py`. |
| Cell patches | `--dataset foci_cell` | Run `python prepare_foci_cell.py --raw-root /path/to/Foci_Dataset` on the folder containing `cell_Pic1/` and `Xml_cell/`. |

Cell preparation defaults to five validation images and split seed 42. With the
50-image source this creates 45 training images and five validation images. XML
class `cell` maps to YOLO class `0` (`cell_patch`). The `.done` manifest records
source hashes, image membership and box counts. Validation box previews are
saved under `data/foci_cell/_check/`. On Kaggle, the default output is
`/kaggle/temp/foci_cell/`, matching the dataset loader.

```bash
python prepare_foci_cell.py --raw-root /path/to/Foci_Dataset --val-size 5 --seed 42
python train.py --dataset foci_cell --aug A --epochs 200 --seed 42 --mrfp none
python evaluate.py --dataset foci_cell --aug A --seeds 42 --mrfp none --ckpt both
```

For dense MRFP experiments, select `plain-conv`, `plain-conv-sac` or `single-conv`
in the same cell training command. Use that same variant for evaluation. The
`vit_foci.py` convenience runner selects the virus task; cell experiments use
`train.py` and `evaluate.py` with `--dataset foci_cell`.

For a fresh Kaggle session, use the [complete cell ablation setup](docs/MRFP_ABLATIONS.md#complete-kaggle-cell-run).
It finds the attached raw data, prepares `/kaggle/temp/foci_cell`, then trains
the four variants over three seeds and evaluates their checkpoints.

Each label row contains class ID `0` and normalized box center coordinates, width and height. `prepare_foci.py` defaults to 15 validation images. The `vit_foci.py` runner defaults to 14 when preparing data itself. Use `--val-size 15` with the runner to match the reported count, or `--skip-prepare` to reuse the exact prepared split. On Kaggle, the loader reads prepared datasets from `/kaggle/temp/`.

Train and compare the virus baseline and combined model for one seed:

```bash
python train.py --dataset foci --aug A --epochs 200 --seed 42
python train.py --dataset foci --aug A --epochs 200 --seed 42 --sac-rates 1,3,5
python evaluate.py --dataset foci --aug A --seeds 42 --configs none 1,3,5 --split val --ckpt best
```

For the cell task, use `--dataset foci_cell` in training and evaluation. For a final-epoch comparison, select `--ckpt last`. To visualize a trained cell baseline's predictions:

```bash
python inference.py --dataset foci_cell --aug A --seed 42 --ckpt last path/to/cell-image.png
```

## MRFP paper ablations

The current neck experiments use the same plain ViT backbone with **normal (dense) convolutions**. `plain-conv` uses channel-split 3 × 3 and 5 × 5 branches. `plain-conv-sac` adds learned dilation 1/3 switches inside those neck branches. `single-conv` uses one normal 3 × 3 per pyramid level with the same output projection. Every spatial branch uses `groups=1`. These CoMer-style variants are separate from the original residual `full` neck evaluated in the new cell result above; the new result does not validate these variants.

Preview the three-seed comparison on prepared FOCI data:

```bash
python vit_foci.py --skip-prepare --aug A --epochs 200 --seeds 42 43 44 \
  --mrfp-variants plain-conv plain-conv-sac single-conv --dry-run
```

Remove `--dry-run` to train and evaluate. [Full ablation guide, checkpoint names and compatibility](docs/MRFP_ABLATIONS.md)

## Sources

The microscopy data and task definitions are credited to A. Singh et al., *Automatic Detection of Virus Infection Patterns in Foci Images Using Switchable Convolutions* (IEEE Access, 2024), and A. Singh and S. Mukherjee, [*Detection of Virus and Small Cell Patches in Foci Images Using Switchable Convolution and Feature Pyramid Networks*](https://arxiv.org/abs/2605.22290).

Historical report results are from Granth Agarwal's *A Vision Transformer YOLOv8 Framework with Switchable Atrous Convolution for Multi-Scale Object Detection* (November 2026), especially Tables 6–8. Earlier general detection experiments are archived in [VOC and African Wildlife benchmarks](docs/LEGACY_BENCHMARKS.md).

For separate cell notebook reproduction and existing FOCI settings, see [isolated run profiles](docs/ISOLATED_RUNS.md). Each profile uses separate checkpoint/log/report names without changing dataset preparation.
