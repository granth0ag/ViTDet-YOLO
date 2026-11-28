# Cell detection: original dense residual MRFP

## Source and scope

Source: [VIT_FOCI on Kaggle, version 42](https://www.kaggle.com/code/granthag/vit-foci?scriptVersionId=374856046), kernel version ID `374856046`, published November 27, 2026; saved outputs inspected November 28, 2026. The run completed successfully. Values below are transcribed from its displayed evaluation tables, not a newly executed benchmark.

This is **one training seed (42), 200 epochs, validation evaluation**. It compares a plain ViT backbone plus SFP/YOLO head with the same backbone plus the original residual MRFP neck. It does not establish multi-seed consistency, test-set performance, or statistical significance. Historical virus and cell results remain separate in [EXPERIMENTS.md](EXPERIMENTS.md).

## Architecture and settings

- Baseline: `--mrfp none`; original MRFP: `--mrfp full`; no `--sac-rates` in either run.
- MRFP: 128→64 channel reduction; GELU; two dense 3×3 branches at dilation 1/3; spatial softmax switching from 5×5 average-pooled features; GELU and 64→128 expansion; residual `x + gamma * output`, with scalar gamma initialized to 0.1. MRFP weights are shared across pyramid levels. There is no BatchNorm inside this MRFP block.
- This is distinct from CoMer-style `plain-conv`, `plain-conv-sac`, and `single-conv`.
- Dataset: `foci_cell`, source `cell_Pic1` / `Xml_cell`, class `cell` mapped to `cell_patch` / ID 0; split seed 42, 45 training images / 4,644 boxes and 5 validation images / 696 boxes.
- Input 640×640; P3/P4/P5; batch size 4; augmentation A; 2 workers.
- Backbone LR 1e-4; neck/head LR 1e-3; AdamW, weight decay 0.05; drop path 0.1; cosine schedule over 200 epochs; gradient clipping 10; CUDA mixed-precision training.
- Validation at epoch 1, every 5 epochs and epoch 200. Best means highest validation AP@0.50:0.95, not highest AP50. Baseline best is epoch 190; MRFP best is epoch 200.
- Standalone evaluation: confidence 0.01, NMS IoU 0.65, full precision; standard TorchMetrics COCO AP and AR100, plus AP25. The crowded-image detection-limit warning remains relevant; no limits were changed for this result.
- Run name: `cell-notebook-e200`. Launcher: `python run_isolated.py --profile cell-notebook`.

## Results

All metrics are percentages, rounded to two decimal places in the source display.

| Checkpoint | Variant | AP25 | AP50 | AP75 | COCO AP | AP small | AR100 |
|---|---|---:|---:|---:|---:|---:|---:|
| Best | none | 53.51 | 45.30 | 4.27 | 15.94 | 15.94 | 20.95 |
| Best | full | 53.75 | 47.59 | 5.19 | 15.99 | 15.99 | 20.72 |
| Last (200) | none | 53.45 | 45.29 | 4.06 | 15.64 | 15.64 | 20.68 |
| Last (200) | full | 53.75 | 47.59 | 5.19 | 15.99 | 15.99 | 20.72 |

The source evaluator reports paired COCO AP changes of **+0.04 pp for best** and **+0.34 pp for last**, computed before display rounding. Subtracting rounded table cells yields +0.05 and +0.35 instead. Preserve the reported deltas rather than treating rounded cells as raw measurements. The displayed `±0.00` for n=1 is not evidence of zero variability; it is omitted here.

At epoch 200, AP50 increases by 2.30 pp and AP75 by 1.13 pp. Best-checkpoint AR100 decreases from 20.95 to 20.72. The positive AP changes are observations from this run, not a claim that every metric improves or that the module generalizes across seeds.

Exact parameter counts from the training log:

| Variant | Total parameters | Neck parameters | MRFP parameters |
|---|---:|---:|---:|
| none | 24,134,803 | 861,888 | 0 |
| full | 24,225,366 | 952,451 | 90,563 |

Timing is not reported: the evaluator skips three warm-up batches, while five validation images at batch size four yield only two batches. Its `n/a` timing is not a zero-latency result.

[Machine-readable rounded transcription](../results/cell_notebook_seed42_kaggle_v42.csv). Original checkpoints and microscopy images are not bundled with this documentation.

## Environment and reproduction limits

The Kaggle output records GPU T4 ×2 availability, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, Ultralytics 8.4.220 and TorchMetrics 1.9.0. The training script uses one CUDA device; the hardware label does not mean distributed two-GPU training.

The earlier Colab reference used different package versions (including PyTorch 2.14.1+cu130, timm 1.0.31 and Ultralytics 8.4.215). Matching profile settings is not a guarantee of identical numerical trajectories across environments. The precise Git commit cloned in the Kaggle run was not printed, so no commit hash is asserted here.

To evaluate existing named checkpoints again, without training:

```bash
python run_isolated.py --profile cell-notebook --eval-only
```

For additional seeds and the combined report, follow [ISOLATED_RUNS.md](ISOLATED_RUNS.md). Keep the prepared split fixed. Additional-seed results are not included in this record.
