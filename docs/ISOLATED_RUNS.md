# Isolated cell reproduction and FOCI experiments

Run these notebook cells from the repository folder after installing its requirements and preparing the appropriate dataset with the existing preparation scripts. This launcher never prepares, repartitions, or modifies data. Existing training/evaluation commands retain their defaults and checkpoint names.

| Profile | Dataset | Neck variants | Backbone LR | Workers | Seeds | Epochs |
|---|---|---|---|---|---|---|
| `cell-notebook` | `foci_cell` | `none`, `full` | 1e-4 | 2 | 42 | 200 |
| `foci-current` | `foci` | `none`, `plain-conv`, `single-conv`, `plain-conv-sac` | 3e-4 | 4 | 42, 43, 44 | 200 |

Both use augmentation A, batch size 4, resolution 640, and the plain ViT backbone. `full` is the original dense residual MRFP with learned switching, gamma=0.1 initialization, and shared weights across pyramid levels. It is different from the CoMer-style `plain-conv-sac`. No architecture code was changed for these profiles.

## Cell: two reproduction runs

```python
!python run_isolated.py --profile cell-notebook --dry-run
!python run_isolated.py --profile cell-notebook
```

The second command trains baseline and original MRFP for 200 epochs each, then evaluates both best and last checkpoints. Evaluation uses confidence 0.01 and full precision, matching notebook training validation. Reports include AP25, AP50, AP75, COCO AP, small AP, and AR100.

After checking reproduction, run the additional seeds without rerunning seed 42:

```python
!python run_isolated.py --profile cell-notebook --seeds 43 44
!python run_isolated.py --profile cell-notebook --seeds 42 43 44 --eval-only
```

## FOCI: existing four dense ablations

```python
!python run_isolated.py --profile foci-current --dry-run
!python run_isolated.py --profile foci-current
```

This runs four variants × three seeds × 200 epochs with current FOCI settings. Standalone evaluation retains confidence 0.001 and CUDA mixed precision. Existing FOCI results remain separate; this command starts fresh named runs, not a resume of old checkpoints. To run just seed 42, add `--seeds 42`.

## Outputs and repeat runs

Checkpoints, TensorBoard logs, and evaluation reports include `_run-cell-notebook-e200` or `_run-foci-current-e200`. Variant and seed remain in each checkpoint name. The launcher checks all planned checkpoints before training and refuses to overwrite them. It stops immediately if a subprocess fails.

Use `--eval-only` to evaluate completed runs again. Use `--run-name cell-repeat-2` for a fresh repetition, and supply the same name when evaluating it. A failed partial training run is not automatically resumed; use a new name or run only the untouched seeds separately. Evaluation reports for the same run name can be regenerated.

Inference accepts the same name:

```python
!python inference.py --dataset foci_cell --aug A --seed 42 --mrfp full --run-name cell-notebook-e200 /path/to/image.jpg
```

Advanced direct training supports `--lr-backbone`, `--num-workers`, and `--run-name`. Their defaults preserve the existing LR=3e-4, workers=4, and original filenames. `evaluate.py` also accepts `--run-name` and optional `--full-precision`.

Matching code settings does not guarantee identical GPU results. Match input files, dependency versions, and checkpoint selection too. The reference notebook recorded timm 1.0.31, ultralytics 8.4.215, and torchmetrics 1.9.0. No packages are installed or changed by this launcher.
