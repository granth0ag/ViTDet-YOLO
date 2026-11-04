
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange


# ============================================================
# 1. CONVOLUTIONAL PATCH EMBEDDING
# ============================================================

class ConvPatchEmbed(nn.Module):
    """
    Replaces the original ViT patch projection.

    Input:  [B, 3, 640, 640]
    Output: [B, 384, 40, 40]

    Four stride-2 convolutions progressively:
    3 -> 32 -> 64 -> 128 -> 384
    """

    def __init__(
        self,
        in_ch=3,
        embed_dim=384,
        widths=(32, 64, 128),
    ):
        super().__init__()

        layers = []
        c = in_ch

        # First three convolutional stages
        for w in widths:
            layers.extend([
                nn.Conv2d(
                    c,
                    w,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(8, w),
                nn.GELU(),
            ])
            c = w

        # Final convolutional stage
        layers.append(
            nn.Conv2d(
                c,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. WINDOW ATTENTION UTILITIES
# ============================================================

def window_partition(x, ws):
    """[B, H, W, C] -> [B * num_windows, ws*ws, C]"""

    return rearrange(
        x,
        "b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c",
        ws1=ws,
        ws2=ws,
    )


def window_unpartition(x, ws, H, W):
    """[B * num_windows, ws*ws, C] -> [B, H, W, C]"""

    return rearrange(
        x,
        "(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c",
        nh=H // ws,
        nw=W // ws,
        ws1=ws,
        ws2=ws,
    )


def run_attention(attn, x):
    """Reuse a timm Attention module's weights. x: [B, N, C]"""

    q, k, v = rearrange(
        attn.qkv(x),
        "b n (three h d) -> three b h n d",
        three=3,
        h=attn.num_heads,
    )

    out = F.scaled_dot_product_attention(q, k, v)

    return attn.proj(
        rearrange(out, "b h n d -> b n (h d)")
    )


# ============================================================
# 3. ORIGINAL ViT DET BACKBONE
# ============================================================

class ViTDetBackbone(nn.Module):

    def __init__(
        self,
        img_size=640,
        window_size=10,
        global_idx=(2, 5, 8, 11),
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
        )

        self.patch_size = 16
        self.embed_dim = self.vit.embed_dim
        self.window_size = window_size
        self.global_idx = set(global_idx)

        g = img_size // self.patch_size

        assert g % window_size == 0, (
            f"grid {g} must be divisible by "
            f"window_size {window_size}"
        )

        # Drop CLS position and retain spatial patch positions
        pos = self.vit.pos_embed[:, 1:, :].detach().clone()

        self.pos_embed = nn.Parameter(
            rearrange(
                pos,
                "1 (h w) c -> 1 h w c",
                h=g,
                w=g,
            )
        )

    def forward(self, x):

        # Original timm patch projection
        x = rearrange(
            self.vit.patch_embed.proj(x),
            "b c h w -> b h w c",
        )

        x = x + self.pos_embed

        B, H, W, C = x.shape

        for i, blk in enumerate(self.vit.blocks):

            y = blk.norm1(x)

            if i in self.global_idx:

                # Global attention
                y = rearrange(y, "b h w c -> b (h w) c")

                y = rearrange(
                    run_attention(blk.attn, y),
                    "b (h w) c -> b h w c",
                    h=H,
                    w=W,
                )

            else:

                # Windowed attention
                y = window_unpartition(
                    run_attention(
                        blk.attn,
                        window_partition(y, self.window_size),
                    ),
                    self.window_size,
                    H,
                    W,
                )

            x = x + blk.drop_path1(blk.ls1(y))

            x = x + blk.drop_path2(
                blk.ls2(
                    blk.mlp(blk.norm2(x))
                )
            )

        x = self.vit.norm(x)

        return rearrange(x, "b h w c -> b c h w")


# ============================================================
# 4. CONVOLUTIONAL ViT DET BACKBONE
# ============================================================

class ViTDetConvStemBackbone(ViTDetBackbone):
    """Same as ViTDetBackbone, but the 16x16 linear patchify is replaced by ConvPatchEmbed."""
    def __init__(self, *args, pe_ckpt=None, **kw):
        super().__init__(*args, **kw)                    # builds timm ViT, pos_embed, etc.
        self.vit.patch_embed.proj = ConvPatchEmbed(embed_dim=self.embed_dim)
        if pe_ckpt:                                      # pretrained arm; None = random init
            sd = torch.load(pe_ckpt, map_location="cpu")
            self.vit.patch_embed.proj.load_state_dict(sd)