from pathlib import Path
from src.model.vit import ViTDetBackbone
from src.model.vit_withConvEmbedding import ViTDetConvStemBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel


def run_tag(dataset, img_size, use_p2, patch_embed, pe_ckpt, aug, seed):
    pe = "pelinear" if patch_embed == "linear" else \
         "peconv-" + (Path(pe_ckpt).stem if pe_ckpt else "scratch")
    return f"{dataset}_{img_size}_{'p2' if use_p2 else 'p3p5'}_{pe}_aug{aug}_s{seed}"


def build_model(img_size=640, num_classes=1, use_p2=False, out_ch=128, window_size=10,
                drop_path_rate=0.0, patch_embed="linear", pe_ckpt=None):
    kw = dict(img_size=img_size, window_size=window_size, drop_path_rate=drop_path_rate)
    if patch_embed == "linear":
        bb = ViTDetBackbone(**kw)
    elif patch_embed == "conv":
        bb = ViTDetConvStemBackbone(**kw, pe_ckpt=pe_ckpt)
    else:
        raise ValueError(f"patch_embed must be 'linear' or 'conv', got {patch_embed}")
    neck = SimpleFeaturePyramid(in_ch=bb.embed_dim, out_ch=out_ch, use_p2=use_p2)
    head = YoloHead(num_classes=num_classes, ch=(out_ch,) * len(neck.strides), strides=neck.strides)
    return DetectionModel(bb, neck, head)