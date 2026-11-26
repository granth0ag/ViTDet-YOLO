"""MRFP experiment names shared by every entry point; no training side effects."""

COMER_MRFP_VARIANTS = ("plain-dw", "plain-dw-sac", "single-dw",
                       "plain-conv", "plain-conv-sac", "single-conv")
MRFP_VARIANTS = ("none", "full", "no-dilation", "no-switch", "no-dilation-no-switch") + COMER_MRFP_VARIANTS

MRFP_DESCRIPTIONS = {
    "none": "SFP only (no MRFP)",
    "full": "Legacy residual MRFP: dense 3x3, dilation 1/3, learned switch",
    "no-dilation": "MRFP w/o dilation: dense 3x3, dilation 1/1, learned switch",
    "no-switch": "MRFP w/o switch: dense 3x3, dilation 1/3, fixed mean",
    "no-dilation-no-switch": "MRFP w/o dilation or switch: dense 3x3, dilation 1/1, fixed mean",
    "plain-dw": "Pure MRFP (ViT-CoMer-style): split-channel DW 3x3/5x5, concat, per-level BN",
    "plain-dw-sac": "ViT-CoMer-style MRFP + neck SAC: DW 3x3/5x5, dilation 1/3, learned switches",
    "single-dw": "Single receptive-field fusion: DW 3x3, dilation 1, same BN and pointwise fusion",
    "plain-conv": "Dense MRFP: split-channel normal 3x3/5x5, concat, per-level BN",
    "plain-conv-sac": "Dense MRFP + neck SAC: normal 3x3/5x5, dilation 1/3, learned switches",
    "single-conv": "Single receptive-field fusion: normal 3x3, dilation 1, same BN and pointwise fusion",
}


def resolve_mrfp(variant=None, sac_rates=None):
    """Omitted flag retains legacy behavior: SAC enables full MRFP."""
    if variant is None:
        return "full" if sac_rates is not None else "none"
    if variant not in MRFP_VARIANTS:
        raise ValueError(f"Unknown MRFP variant {variant!r}; choose from {MRFP_VARIANTS}")
    return variant


def mrfp_suffix(variant=None):
    if variant is None:
        return ""  # exact legacy checkpoint and report names
    return f"_mrfp-{resolve_mrfp(variant)}"


def model_label(sac_rates=None, mrfp_variant=None):
    backbone = "ViT" if sac_rates is None else f"ViT + SAC d={','.join(map(str, sac_rates))}"
    return f"{backbone}; {MRFP_DESCRIPTIONS[resolve_mrfp(mrfp_variant, sac_rates)]}"


def add_mrfp_args(parser, sweep=False):
    group = parser.add_mutually_exclusive_group() if sweep else parser
    group.add_argument("--mrfp", "--mrfp-variant", dest="mrfp", nargs="?", const="full",
                       default=None, choices=MRFP_VARIANTS,
                       help="select the neck independently of backbone SAC; --mrfp alone means full; "
                            "omitting it preserves the original SAC/MRFP pairing")
    if sweep:
        group.add_argument("--mrfp-variants", nargs="+", choices=MRFP_VARIANTS,
                           help="compare these MRFP variants in the supplied order; first is the reference")


def requested_variants(args):
    variants = getattr(args, "mrfp_variants", None)
    return list(dict.fromkeys(variants)) if variants is not None else [args.mrfp]


def mrfp_report_suffix(variants):
    if variants == [None]:
        return ""
    return "_mrfp-" + "_".join(variants)
