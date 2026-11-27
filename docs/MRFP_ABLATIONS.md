# MRFP paper ablations

The paper comparison uses **three dense-convolution necks with the same plain ViT backbone**:

| Experiment | CLI option | Spatial operation inside each pyramid level |
|---|---|---|
| MRFP with normal convolutions | `--mrfp plain-conv` | Split reduced channels into normal 3×3 and 5×5, dilation 1; concatenate |
| MRFP with SAC inside the neck | `--mrfp plain-conv-sac` | Same channel split and kernels; each kernel group has dilation 1/3 branches with a learned softmax switch |
| Single receptive-field fusion | `--mrfp single-conv` | One normal 3×3 over all reduced channels, dilation 1 |

All spatial convolutions use `groups=1`, including both dilation branches inside
SAC. In multi-kernel MRFP, each branch mixes all channels in its half of the
reduced representation. The 1×1 output projection mixes the concatenated halves.

All three retain the same shared 1×1 channel reduction/expansion, per-level
BatchNorm, activation order and pyramid levels. The final 1×1 projection fuses
channels in all three. None adds a residual/gamma inside the MRFP block.
`single-conv` removes the parallel 3×3/5×5 kernel choice, not the existing SFP
pyramid: “single receptive field” refers to the neck's spatial kernel at each level.

For `plain-conv-sac`, the 3×3 group and 5×5 group each use two independently learned
normal convolutions with dilation 1 and 3. A 5×5 average pool followed by a 1×1
switch predicts two spatial gate weights; zero logits start at a 0.5/0.5 mixture.
Switches and spatial filters are independent across pyramid levels. This is a
SAC-style extension of the reference neck, following this project's independent
branch-weight convention. It does not enable SAC in the ViT attention blocks.

Compare dense MRFP with `single-conv` to study multi-kernel versus single-kernel
fusion, and dense MRFP with `plain-conv-sac` to study the added dilation/switching.
Parameter counts differ; reports include total and MRFP-only counts, so these
are not parameter-matched controls. Keep seeds, epochs, input size, augmentation,
splits and backbone fixed across runs.

### Run the three FOCI experiments

The existing runner trains the selected variants and produces one combined
report. Use `--skip-prepare` if the FOCI dataset is already prepared. Preview:

```bash
python vit_foci.py --skip-prepare --aug A --epochs 200 --seeds 42 43 44 \
  --mrfp-variants plain-conv plain-conv-sac single-conv --dry-run
```

Remove `--dry-run` to execute **three variants × three seeds = nine training
runs**, followed by evaluation. Set epochs and augmentation to the same values
as your other experiments. No dataset preparation or training happens in a dry
run. Omit `--skip-prepare` to use the existing preparation/reuse behavior.

For this comparison, leave `--sac-rates` empty: that existing option controls
**backbone** SAC. The `plain-conv-sac` variant itself enables **neck** SAC.
Use explicit variant names; `--mrfp` without a value still selects the older
`full` variant for compatibility.

Single runs and inference use the existing entry points:

```bash
python train.py --dataset foci --aug A --seed 42 --epochs 200 --mrfp plain-conv
python train.py --dataset foci --aug A --seed 42 --epochs 200 --mrfp plain-conv-sac
python train.py --dataset foci --aug A --seed 42 --epochs 200 --mrfp single-conv
python inference.py --dataset foci --aug A --seed 42 --mrfp plain-conv-sac path/to/image.png
```

Evaluate the three experiments together:

```bash
python evaluate.py --dataset foci --aug A --seeds 42 43 44 \
  --mrfp-variants plain-conv plain-conv-sac single-conv
```

Dense MRFP, the first listed configuration, is the paired-delta reference.
With explicit MRFP options, `evaluate.py` defaults to `--configs none` (plain
ViT). The FOCI runner evaluates whichever backbone it trained. `--configs`
continues to mean backbone SAC configurations, not neck variants.

Each explicit variant has a separate checkpoint and TensorBoard suffix, e.g.
`best_foci_640_p3p5_augA_s42_mrfp-plain-conv-sac.pt` and
`runs/foci_640_p3p5_augA_s42_mrfp-single-conv`. Reports include the variant list and
backbone configuration in their filenames. Pass the **same MRFP variant** to
training, evaluation and inference. Train each new architecture; old SAC-backbone
checkpoints should not be relabelled as these neck-only experiments.

### Cell-patch experiments

Prepare the cell images and XML annotations once, then use `foci_cell` for all
training and evaluation runs. The default split uses five validation images
and seed 42. Hold this dataset split fixed while varying training seeds.

```bash
python prepare_foci_cell.py --raw-root /path/to/Foci_Dataset --val-size 5 --seed 42
python train.py --dataset foci_cell --aug A --epochs 200 --seed 42 --mrfp none
python train.py --dataset foci_cell --aug A --epochs 200 --seed 42 --mrfp plain-conv
python train.py --dataset foci_cell --aug A --epochs 200 --seed 42 --mrfp plain-conv-sac
python train.py --dataset foci_cell --aug A --epochs 200 --seed 42 --mrfp single-conv
python evaluate.py --dataset foci_cell --aug A --seeds 42 \
  --mrfp-variants none plain-conv plain-conv-sac single-conv
```

Repeat training with seeds 43 and 44 for the three-seed study, then evaluate with
`--seeds 42 43 44`. Prepared images and labels are reused across every run.
The source root contains `cell_Pic1/` and `Xml_cell/`; on Kaggle supply the mounted
dataset root under `/kaggle/input/`. Outputs default to the loader's cell path.

### Complete Kaggle cell run

Attach the extracted FOCI dataset through Kaggle's **Add Input**. Run this cell
after installing the repository dependencies. Each fresh session prepares its
own `/kaggle/temp/foci_cell` folder before training. Preparation fixes the data
split at seed 42, while training uses seeds 42, 43 and 44.

```python
import os
from pathlib import Path

os.chdir("/kaggle/working/ViTDet-YOLO")
if not Path("prepare_foci_cell.py").is_file():
    raise RuntimeError("Update the repository to include prepare_foci_cell.py")

roots = sorted({
    folder.parent.resolve()
    for folder in Path("/kaggle/input").rglob("cell_Pic1")
    if folder.is_dir() and (folder.parent / "Xml_cell").is_dir()
})
if len(roots) != 1:
    raise RuntimeError(
        f"Expected one dataset containing cell_Pic1 and Xml_cell, found {roots}. "
        "Attach the extracted FOCI dataset through Add Input."
    )
raw_root = str(roots[0])

!python prepare_foci_cell.py --raw-root "{raw_root}" --output /kaggle/temp/foci_cell --val-size 5 --seed 42
if _exit_code != 0:
    raise RuntimeError("Cell dataset preparation failed")

for split in ("train", "val"):
    folder = Path("/kaggle/temp/foci_cell/images") / split
    print(f"{split}: {len(list(folder.iterdir()))} prepared images", flush=True)

for variant in ("none", "plain-conv", "single-conv", "plain-conv-sac"):
    for seed in (42, 43, 44):
        print(f"\nCell training | {variant} | seed={seed} | epochs=200", flush=True)
        !python train.py --dataset foci_cell --aug A --epochs 200 --seed {seed} --mrfp {variant}
        if _exit_code != 0:
            raise RuntimeError(f"Training failed: {variant}, seed {seed}")

!python evaluate.py --dataset foci_cell --aug A --seeds 42 43 44 --configs none --mrfp-variants none plain-conv single-conv plain-conv-sac --ckpt both
if _exit_code != 0:
    raise RuntimeError("Cell evaluation failed")

print("All 12 cell runs and combined evaluation completed.")
```

### Dense and depthwise runs

The earlier `plain-dw`, `plain-dw-sac` and `single-dw` flags retain their depthwise
operations and checkpoint layouts. Compare each against its corresponding
`plain-conv`, `plain-conv-sac` or `single-conv` variant to study dense channel
mixing. Channel widths, kernel sizes, dilation rates, switches, normalization,
activations and fusion follow the corresponding DW variant. Dense filters add
parameters and computation, which should be included when interpreting results.

The new dense experiments require fresh training and have distinct checkpoint,
TensorBoard and report names. Use the same variant when evaluating or inferring.
The `full` flag still selects the earlier residual dense 3×3 dilation 1/3 block.

### What is taken from ViT-CoMer

The official MRFP uses **both 3×3 and 5×5**, on separate halves of the reduced
channels. Its `MultiDWConv` has independent spatial filters and BatchNorm for each
of P3, P4 and P5. The shared projections reduce channels by half and restore them.
The order is projection → split-channel DW3/DW5 → concatenate → BatchNorm/GELU →
GELU → projection (the second GELU is present in the official MRFP forward).

`plain-dw` implements that module computation on the existing SFP feature maps,
using equivalent 1×1 projections instead of flattening into tokens. It retains
this project's PyTorch module initialization; it does not import ViT-CoMer's
whole-backbone initialization routine, CNN branch or CTI interactions. No new
MMDetection/MMCV dependency is introduced. This is a **ViT-CoMer-style MRFP module
reference in this detector**, not a reproduction of the full ViT-CoMer model.
If P2 is manually enabled, the same operation is extended to the fourth level.

Sources: [CVPR paper, Section 3.2 and Figure 4](https://openaccess.thecvf.com/content/CVPR2024/papers/Xia_ViT-CoMer_Vision_Transformer_with_Convolutional_Multi-scale_Feature_Interaction_for_Dense_CVPR_2024_paper.pdf)
and [official MRFP / MultiDWConv code](https://github.com/Traffic-X/ViT-CoMer/blob/main/detection/mmdet_custom/models/backbones/comer_modules.py#L81-L164).

### Compatibility and earlier controls

Commands and model construction calls that omit the MRFP option retain the
existing behavior: plain ViT + SFP with no SAC rates, or SAC backbone + the
original residual MRFP when rates are supplied. Their checkpoint names and
state-dict layouts are unchanged. `--mrfp-variant` remains an alias for `--mrfp`.

Earlier options remain available: `none` (SFP only), `full` (the original dense
3×3 residual MRFP with dilation 1/3 and a learned switch), `no-dilation`,
`no-switch`, and `no-dilation-no-switch`. Those controls operate on the older
residual MRFP; they are separate from the three paper experiments above.

Dataset preparation, labels, splits, augmentation, losses, optimizer settings
and prediction limits are unchanged. Training accepts the existing optional third
test loader while training/validating on the first two splits. No new FOCI sample
limit is applied. The standalone `vit_foci.ipynb` remains unchanged; use the Python
entry points above for these new variants.

CPU regression tests (synthetic data, no pretrained-weight or dataset downloads):

```bash
python -m unittest discover -s tests -v
```
