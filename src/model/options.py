from math import isfinite


def add_training_options(parser):
    parser.add_argument("--nwd-weight", type=float, default=0.0,
                        help="extra NWD term inside the YOLO box loss (0 disables)")
    parser.add_argument("--nwd-scale", type=float, default=1.6,
                        help="NWD distance scale in feature-grid cells")
    parser.add_argument("--aux-center-weight", type=float, default=0.0,
                        help="auxiliary center heatmap loss weight (0 disables)")


def option_suffix(args):
    weights = (args.nwd_weight, args.aux_center_weight)
    if any(not isfinite(w) or w < 0 for w in weights):
        raise ValueError("Loss weights must be finite and nonnegative")
    if not isfinite(args.nwd_scale) or args.nwd_scale <= 0:
        raise ValueError("--nwd-scale must be finite and positive")
    suffix = ""
    if args.nwd_weight:
        suffix += f"_nwd{args.nwd_weight}_scale{args.nwd_scale}"
    if args.aux_center_weight:
        suffix += f"_center{args.aux_center_weight}"
    return suffix
