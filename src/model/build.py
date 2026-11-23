from src.model.vit import ViTDetBackbone

from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel
from src.model.vit_sac import ViTDetSACBackbone  
from src.model.mrfp import MRFPNeck  

def run_tag(dataset, img_size, use_p2, aug, seed, sac=None, *, nwd=False, aux_center=False):
    tag = f"{dataset}_{img_size}_{'p2' if use_p2 else 'p3p5'}_aug{aug}_s{seed}"
    if sac:
        tag += f"_sac{''.join(map(str, sac))}"
    if nwd:
        tag += "_nwd"
    if aux_center:
        tag += "_center"
    return tag


def build_model(img_size=640, num_classes=1, use_p2=False, out_ch=128,
                window_size=10, drop_path_rate=0.0, sac_rates=None, aux_center=False, pretrained=True):
    kw = dict(img_size=img_size, window_size=window_size, drop_path_rate=drop_path_rate, pretrained=pretrained)

    if sac_rates is not None:
        bb = ViTDetSACBackbone(**kw, sac_rates=sac_rates)
    else:
        bb = ViTDetBackbone(**kw)

    neck = SimpleFeaturePyramid(in_ch=bb.embed_dim, out_ch=out_ch, use_p2=use_p2)

    if sac_rates is not None:
        neck = MRFPNeck(neck, ch=out_ch)
    head = YoloHead(num_classes=num_classes, ch=(out_ch,) * len(neck.strides), strides=neck.strides)
    return DetectionModel(bb, neck, head, center_channels=out_ch if aux_center else None)
