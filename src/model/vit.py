import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


def window_partition(x, ws):
    """[B, H, W, C] -> [B * num_windows, ws*ws, C]"""
    return rearrange(x, "b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c", ws1=ws, ws2=ws)


def window_unpartition(x, ws, H, W):
    """[B * num_windows, ws*ws, C] -> [B, H, W, C]"""
    return rearrange(x, "(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c",
                     nh=H // ws, nw=W // ws, ws1=ws, ws2=ws)


def run_attention(attn, x):
    """Reuse a timm Attention module's weights. x: [B, N, C]"""
    q, k, v = rearrange(attn.qkv(x), "b n (three h d) -> three b h n d",
                        three=3, h=attn.num_heads)
    out = F.scaled_dot_product_attention(q, k, v)
    return attn.proj(rearrange(out, "b h n d -> b n (h d)"))


class ViTDetBackbone(nn.Module):
    def __init__(self, img_size=640, window_size=10, global_idx=(2, 5, 8, 11)):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0,
                                 img_size=img_size, drop_path_rate=0.1)
        self.patch_size = 16
        self.embed_dim = self.vit.embed_dim
        self.window_size = window_size
        self.global_idx = set(global_idx)

        g = img_size // self.patch_size # these are the number of number of patches (stride-16) along each axis, e.g. 320/16=20
        assert g % window_size == 0, f"grid {g} must be divisible by window_size {window_size}"

        # drop the CLS position, keep patch positions as a [1, g, g, C] grid
        pos = self.vit.pos_embed[:, 1:, :].detach().clone()
        self.pos_embed = nn.Parameter(rearrange(pos, "1 (h w) c -> 1 h w c", h=g, w=g))

    def forward(self, x):
        x = rearrange(self.vit.patch_embed.proj(x), "b c h w -> b h w c")
        x = x + self.pos_embed
        B, H, W, C = x.shape

        for i, blk in enumerate(self.vit.blocks):
            y = blk.norm1(x)
            if i in self.global_idx:                           # cross-window propagation
                y = rearrange(y, "b h w c -> b (h w) c")
                y = rearrange(run_attention(blk.attn, y), "b (h w) c -> b h w c", h=H, w=W)
            else:                                              # windowed attention
                y = window_unpartition(
                    run_attention(blk.attn, window_partition(y, self.window_size)),
                    self.window_size, H, W)
            x = x + blk.drop_path1(blk.ls1(y))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))

        x = self.vit.norm(x)
        return rearrange(x, "b h w c -> b c h w")