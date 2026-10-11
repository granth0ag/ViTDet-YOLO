from src.model.vit import ViTDetBackbone
from src.model.fpn import SimpleFeaturePyramid
from src.model.detector import YoloHead, DetectionModel


def build_model(img_size=640, num_classes=4, use_p2=False, out_ch=128,
                window_size=10, drop_path_rate=0.0):
    backbone = ViTDetBackbone(img_size=img_size, window_size=window_size, drop_path_rate=drop_path_rate)
    neck = SimpleFeaturePyramid(in_ch=backbone.embed_dim, out_ch=out_ch, use_p2=use_p2)
    head = YoloHead(num_classes=num_classes, ch=(out_ch,) * len(neck.strides), strides=neck.strides)
    return DetectionModel(backbone, neck, head)