# ViTDet-YOLO for FOCI and cell-patch detection

This PyTorch detector draws bounding boxes around **virus foci** and **small cell patches** in microscopy images. These are two separate, single-class detection tasks. The project studies whether giving a Vision Transformer access to different local neighborhoods improves box localization.

On the virus-patch validation split, adding switchable atrous convolution (SAC) to the backbone and multi-receptive-field refinement (MRFP) to the neck raised mean AP@0.50 from **78.84% to 80.47%** across three matched training seeds. Cell-patch localization was harder. Across the four cell models, AP@0.75 ranged from **3.21% to 4.34%** on five validation images.

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

| Model | mAP@0.25 | mAP@0.50 | mAP@0.75 | mAP@0.50:0.95 | AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain ViT | 53.61 | 45.57 | 3.94 | 15.33 | 20.50 |
| SAC in the ViT backbone | 53.66 | 43.74 | 4.34 | 15.28 | 20.43 |
| SAC in the YOLOv8 head | 52.25 | 42.28 | 3.21 | 13.34 | 18.51 |
| Backbone SAC + dense MRFP neck | 54.89 | 46.42 | 3.85 | 15.48 | 20.60 |

The first three rows use final-epoch evaluations. The combined row uses the latest pre-final validation snapshot reported in Table 8. It reached 46.42% mAP@0.50 and 15.48% mAP@0.50:0.95. Backbone SAC had the highest mAP@0.75 (4.34%). [Checkpoint details](docs/EXPERIMENTS.md#cell-patch-detection)

## Run a microscopy experiment

Install the project dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

| Task | Dataset flag | Prepare the data |
|---|---|---|
| Virus foci | `--dataset foci` | Place the source folders `JPEGImages/JPEGImages/` and `Virus_anno/Annotations/` under `data/foci_raw/`, then run `python prepare_foci.py`. |
| Cell patches | `--dataset foci_cell` | Place YOLO-format data under `data/foci_cell/` with `images/train`, `images/val`, `labels/train` and `labels/val`. |

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

The current neck experiments use the same plain ViT backbone to compare three designs. `plain-dw` uses ViT-CoMer-style depthwise 3 × 3 and 5 × 5 branches. `plain-dw-sac` adds learned dilation switches inside those neck branches. `single-dw` uses one depthwise 3 × 3 per pyramid level with the same output projection. The scores above belong to the earlier experiments.

Preview the three-seed comparison on prepared FOCI data:

```bash
python vit_foci.py --skip-prepare --aug A --epochs 200 --seeds 42 43 44 \
  --mrfp-variants plain-dw plain-dw-sac single-dw --dry-run
```

Remove `--dry-run` to train and evaluate. [Full ablation guide, checkpoint names and compatibility](docs/MRFP_ABLATIONS.md)

## Sources

The microscopy data and task definitions are credited to A. Singh et al., *Automatic Detection of Virus Infection Patterns in Foci Images Using Switchable Convolutions* (IEEE Access, 2024), and A. Singh and S. Mukherjee, [*Detection of Virus and Small Cell Patches in Foci Images Using Switchable Convolution and Feature Pyramid Networks*](https://arxiv.org/abs/2605.22290).

Results are from Granth Agarwal's *A Vision Transformer YOLOv8 Framework with Switchable Atrous Convolution for Multi-Scale Object Detection* (November 2026), especially Tables 6–8. Earlier general detection experiments are archived in [VOC and African Wildlife benchmarks](docs/LEGACY_BENCHMARKS.md).
