# src/model/build.py
from src.model.vit import ViTDetBackbone
from src.model.vit_withconv import ViTDetConvBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel

BACKBONES = ("vit", "conv")


def run_tag(dataset, img_size, use_p2, backbone, aug):
    """Checkpoint / TensorBoard / outputs ka naam. train.py, evaluate.py, inference.py teeno yahi use karein."""
    return (f"{dataset}_{img_size}_{'p2' if use_p2 else 'p3p5'}"
            f"{'_conv' if backbone == 'conv' else ''}_aug{aug}")


def build_model(img_size=640, num_classes=4, use_p2=False, out_ch=128,
                window_size=10, drop_path_rate=0.0,
                backbone="vit", act_after_dw=False, conv_blocks=None):
    """
    backbone = "vit"  -> original ViTDetBackbone (vit.py)
    backbone = "conv" -> ViTDetConvBackbone (vit_withconv.py): LocalViT-style conv MLP
                         act_after_dw, conv_blocks sirf "conv" mein use hote hain
    """
    if backbone == "vit":
        bb = ViTDetBackbone(img_size=img_size, window_size=window_size,
                            drop_path_rate=drop_path_rate)
    elif backbone == "conv":
        bb = ViTDetConvBackbone(img_size=img_size, window_size=window_size,
                                drop_path_rate=drop_path_rate,
                                act_after_dw=act_after_dw, conv_blocks=conv_blocks)
    else:
        raise ValueError(f"backbone '{backbone}' galat hai, {BACKBONES} mein se chun")

    neck = SimpleFeaturePyramid(in_ch=bb.embed_dim, out_ch=out_ch, use_p2=use_p2)
    head = YoloHead(num_classes=num_classes, ch=(out_ch,) * len(neck.strides), strides=neck.strides)
    return DetectionModel(bb, neck, head)