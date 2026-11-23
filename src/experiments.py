"""CLI options shared by the cell detection ablations."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class InferenceOptions:
    sahi: bool = False
    soft_nms: bool = False
    tile_size: int | None = None
    tile_overlap: float = 0.25
    soft_nms_sigma: float = 0.5


# The training flags select the checkpoint. Inference flags reuse that checkpoint.
ABLATIONS = {
    "e0": (False, False, False, False),
    "e1": (True, False, False, False),
    "e2": (False, True, False, False),
    "e3": (False, False, True, False),
    "e4": (False, False, False, True),
    "all": (True, True, True, True),
}


def add_training_options(parser):
    parser.add_argument("--nwd-loss", action="store_true", help="add NWD to the CIoU box loss")
    parser.add_argument("--aux-center-loss", action="store_true", help="train an auxiliary center-offset head")


def add_inference_options(parser):
    parser.add_argument("--sahi", action="store_true", help="run overlapping SAHI crops and a full-image pass")
    parser.add_argument("--soft-nms", action="store_true", help="use Gaussian Soft-NMS")
    parser.add_argument("--tile-size", type=int, help="square crop size in source pixels (default: half the long side)")
    parser.add_argument("--tile-overlap", type=float, default=0.25, help="fraction of overlap between adjacent crops")
    parser.add_argument("--soft-nms-sigma", type=float, default=0.5)


def validate_experiment_dataset(parser, dataset, enabled):
    if enabled and dataset not in {"foci_cell", "voc", "wildlife"}:
        parser.error("experimental options are supported on foci_cell, voc and wildlife")


def inference_options(parser, args):
    if args.tile_size is not None and args.tile_size < 1:
        parser.error("--tile-size must be positive")
    if not 0 <= args.tile_overlap < 1:
        parser.error("--tile-overlap must be in [0, 1)")
    if not isfinite(args.soft_nms_sigma) or args.soft_nms_sigma <= 0:
        parser.error("--soft-nms-sigma must be positive and finite")
    return InferenceOptions(args.sahi, args.soft_nms, args.tile_size, args.tile_overlap, args.soft_nms_sigma)
