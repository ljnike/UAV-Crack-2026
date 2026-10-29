"""Crack-segmentation model zoo.

Design goals (driven by the data): thin cracks (~0.4-2 % of pixels, often only
1-4 px wide) must not be destroyed by downsampling.  Every architecture here
therefore *keeps as much resolution as possible* and up-samples back to the
exact input resolution with `align_corners=False` bilinear interpolation.

Components follow the MMSegmentation design language (`mmseg/models/...`):
`ASPPHead`-style multi-scale context, `SegformerHead`-style light fusion,
`DiceLoss`/`FocalLoss`/`LovaszLoss` and slide+TTA inference.  These are
re-implementations on top of plain torch/torchvision - no mmcv build needed.

All models return a single logit map with shape (N, 1, H, W).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- basics


def conv_bn_act(in_ch: int, out_ch: int, k: int = 3, s: int = 1, d: int = 1,
                act: bool = True, groups: int = 1) -> nn.Sequential:
    pad = d * (k // 2)
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad, dilation=d,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def ds_conv_bn_act(in_ch: int, out_ch: int, k: int = 3, d: int = 1) -> nn.Sequential:
    """Depthwise-separable 3x3 conv (MobileNet-style), as used by MMSeg's
    `DepthwiseSeparableASPPHead`.  ~8x cheaper than a dense 3x3 at the same
    width, which matters a lot at full 672x378 resolution."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, k, padding=d * (k // 2), dilation=d, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class SCSEModule(nn.Module):
    """Spatial + channel squeeze-and-excitation (Roy et al., 'Concurrent SE')."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        c = max(1, ch // r)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, ch, 1), nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


class GatedAttentionBlock(nn.Module):
    """Gated channel-spatial attention refinement.

    Implements the multiplicative-gating pattern used by gated interactive /
    gated attention units: a *global* context branch (channel gate, sigmoid)
    and a *local* depthwise branch (spatial gate, sigmoid) produce two
    independent gates whose product modulates the input residual:

        g_c = sigmoid(W2 . relu(W1 . GAP(x)))          # global channel gate
        g_s = sigmoid(DWConv(x))                        # local spatial gate
        y   = x + W_o( x * g_c * g_s )

    `mode` selects which gates are active, so each variant can be ablated.
    """

    def __init__(self, ch: int, r: int = 8, mode: str = "both"):
        super().__init__()
        assert mode in ("both", "channel", "spatial")
        self.mode = mode
        c = max(1, ch // r)
        self.gate_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, c, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, ch, 1), nn.Sigmoid(),
        )
        self.gate_s = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "channel":
            g = self.gate_c(x)
        elif self.mode == "spatial":
            g = self.gate_s(x)
        else:
            g = self.gate_c(x) * self.gate_s(x)
        return x + self.proj(x * g)


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling (MMSeg `ASPPHead` core)."""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations: Sequence[int] = (1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d == 1:
                self.branches.append(conv_bn_act(in_ch, out_ch, k=1))
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                    )
                )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.gap(x), size=x.shape[-2:], mode="bilinear",
                          align_corners=False)
        feats.append(g)
        return self.project(torch.cat(feats, dim=1))


class GeodesicStem(nn.Module):
    """Two stride-1 convolutions before any downsampling.

    Preserves 1-2 px wide cracks that a stride-2 stem would erase.  A 1x1
    shortcut makes the block usable as the network's first stage.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn_act(in_ch, out_ch, 3, 1),
            conv_bn_act(out_ch, out_ch, 3, 1),
        )
        self.short = conv_bn_act(in_ch, out_ch, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.short(x))


# ------------------------------------------------------------------- encoder


class ResNetEncoder(nn.Module):
    """torchvision ResNet truncated into 5 levels: /1, /4, /8, /16, /32."""

    def __init__(self, arch: str = "resnet34", pretrained: bool = True,
                 replace_stem: bool = True, stem_ch: int = 32):
        super().__init__()
        import torchvision.models as tvm

        table = {
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
            "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, [64, 256, 512, 1024, 2048]),
        }
        if arch not in table:
            raise ValueError(f"unsupported encoder {arch}")
        ctor, weights, widths = table[arch]
        net = ctor(weights=weights if pretrained else None)

        self.arch = arch
        self.widths = widths
        self.replace_stem = replace_stem
        if replace_stem:
            # Keep the pretrained weights of layer1..layer4 but re-learn the stem.
            # `layer1` expects `widths[0]` input channels, so the geodesic stem must
            # emit exactly that many channels (a mismatch here is a silent crash).
            self.stem = GeodesicStem(3, widths[0])
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = net.layer1          # /4
            self.stem_out_ch = widths[0]
        else:
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
            self.pool = net.maxpool
            self.layer1 = net.layer1
            self.stem_out_ch = 64
        self.layer2 = net.layer2              # /8
        self.layer3 = net.layer3              # /16
        self.layer4 = net.layer4              # /32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c0 = self.stem(x)                     # /1
        c1 = self.layer1(self.pool(c0))       # /4
        c2 = self.layer2(c1)                  # /8
        c3 = self.layer3(c2)                  # /16
        c4 = self.layer4(c3)                  # /32
        return [c0, c1, c2, c3, c4]


# ------------------------------------------------------------------- decoder


class UpBlock(nn.Module):
    """Upsample -> concat skip -> fusion (U-Net++ / FPN style).

    `style='dense'` uses two 3x3 convolutions; `style='ds'` uses depthwise
    separable convolutions with dilation, which is ~8x cheaper at the full
    decoder resolution and is what MMSeg's DepthwiseSeparable heads do.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, attn: bool = True,
                 gat_mode: str = "both", style: str = "dense", dilation: int = 1):
        super().__init__()
        self.attn = SCSEModule(skip_ch) if attn else nn.Identity()
        self.style = style
        if style == "ds":
            self.conv1 = ds_conv_bn_act(in_ch + skip_ch, out_ch, d=dilation)
            self.conv2 = ds_conv_bn_act(out_ch, out_ch, d=dilation)
        else:
            self.conv1 = conv_bn_act(in_ch + skip_ch, out_ch)
            self.conv2 = conv_bn_act(out_ch, out_ch)
        self.gat = GatedAttentionBlock(out_ch, mode=gat_mode) if attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.attn(skip)], dim=1)
        return self.gat(self.conv2(self.conv1(x)))


class CrackUNet(nn.Module):
    """High-resolution U-Net for thin-crack segmentation.

    Extra features vs. a vanilla U-Net:
      * geodesic (stride-1) stem keeps the native resolution;
      * SCSE-gated skips;
      * optional ASPP multi-scale bottleneck;
      * optional gated-attention refinement at the two highest resolutions;
      * output head restores the *exact* input size (no stride-2 truncation).
    """

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True,
                 decoder_ch: Sequence[int] = (128, 96, 64, 48), aspp: bool = True,
                 attn: bool = True, gat_mode: str = "both", out_channels: int = 1,
                 deep_supervision: bool = False, plain_stem: bool = False,
                 style: str = "dense", dilate_high_res: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained,
                                     replace_stem=not plain_stem, stem_ch=32)
        w = self.encoder.widths
        self.aspp = ASPP(w[4], 256) if aspp else None
        d0, d1, d2, d3 = decoder_ch
        # the top two decoder stages run at 1/4 and 1/1 of the input: they are by
        # far the most expensive, so they can use dilated separable convs
        s_hi, s_lo = ("ds", "ds") if style == "ds" else ("dense", "dense")
        dil = 2 if (style == "ds" and dilate_high_res) else 1
        self.dec4 = UpBlock(self.aspp_out_ch(w[4]) if aspp else w[4], w[3], d0, attn,
                            gat_mode, style=s_lo, dilation=1)
        self.dec3 = UpBlock(d0, w[2], d1, attn, gat_mode, style=s_lo, dilation=1)
        self.dec2 = UpBlock(d1, w[1], d2, attn, gat_mode, style=s_hi, dilation=dil)
        self.dec1 = UpBlock(d2, self.encoder.stem_out_ch, d3, attn, gat_mode,
                            style=s_hi, dilation=dil)
        self.head = nn.Sequential(
            conv_bn_act(d3, 32),
            nn.Conv2d(32, out_channels, 1),
        )
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux1 = nn.Conv2d(d0, out_channels, 1)
            self.aux2 = nn.Conv2d(d1, out_channels, 1)

    @staticmethod
    def aspp_out_ch(_: int) -> int:
        return 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        c0, c1, c2, c3, c4 = self.encoder(x)
        b = self.aspp(c4) if self.aspp is not None else c4
        d = self.dec4(b, c3)
        d = self.dec3(d, c2)
        d = self.dec2(d, c1)
        d = self.dec1(d, c0)
        out = self.head(d)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a ResNet backbone (MMSeg `DepthwiseSeparableASPPHead`-like).

    The low-level branch uses the /4 feature, so cracks stay reasonably sharp.
    """

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True,
                 low_ch: int = 48, aspp_ch: int = 256, out_channels: int = 1,
                 se: bool = False, attn: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained, replace_stem=se, stem_ch=32)
        w = self.encoder.widths
        self.reduced_stem = se
        self.aspp = ASPP(w[4], aspp_ch)
        self.low_proj = conv_bn_act(w[1], low_ch, k=1)
        self.fuse = nn.Sequential(
            conv_bn_act(aspp_ch + low_ch, 256),
            conv_bn_act(256, 128),
            nn.Conv2d(128, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        _, c1, _, _, c4 = self.encoder(x)
        b = self.aspp(c4)
        b = F.interpolate(b, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([b, self.low_proj(c1)], dim=1))
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


class SegFormerHead(nn.Module):
    """Light all-MLP fusion head from SegFormer (MMSeg `SegformerHead`).

    `in_channels` must be a *per-level* sequence (e.g. (64, 128, 320, 512)).
    Passing a single int is accepted and expanded, which keeps the factory
    tolerant of a mis-specified width.
    """

    def __init__(self, in_channels: Sequence[int], embed_dim: int = 256,
                 out_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, 1, bias=False),
                          nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            conv_bn_act(embed_dim * len(in_channels), embed_dim, k=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, out_channels, 1),
        )

    def forward(self, feats: Sequence[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        target = feats[0].shape[-2:]
        outs = [F.interpolate(p(f), size=target, mode="bilinear", align_corners=False)
                for p, f in zip(self.proj, feats)]
        y = self.fuse(torch.cat(outs, dim=1))
        if y.shape[-2:] != out_size:
            y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
        return y


class MiTEncoder(nn.Module):
    """SegFormer MiT (mix transformer) encoder built on torchvision's ViT
    patch-embed + our own efficient self-attention blocks.

    Implemented from the SegFormer paper (Xie et al., NeurIPS 2021) rather than
    vendored, so it runs without mmcv/timm mismatches.
    """

    def __init__(self, variant: str = "b2"):
        super().__init__()
        cfg = {
            "b0": ([32, 64, 160, 256], [2, 2, 2, 2], [8, 4, 2, 1], [1, 2, 5, 8]),
            "b1": ([64, 128, 320, 512], [2, 2, 2, 2], [8, 4, 2, 1], [1, 2, 5, 8]),
            "b2": ([64, 128, 320, 512], [3, 4, 6, 3], [8, 4, 2, 1], [1, 2, 5, 8]),
            "b3": ([64, 128, 320, 512], [3, 4, 18, 3], [8, 4, 2, 1], [1, 2, 5, 8]),
        }[variant]
        dims, depths, sr, heads = cfg
        self.dims = dims
        self.patch_embed = nn.ModuleList()
        for i, (d, s) in enumerate(zip(dims, [4, 2, 2, 2])):
            in_ch = 3 if i == 0 else dims[i - 1]
            self.patch_embed.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, d, kernel_size=s, stride=s,
                              padding=s // 2 if s == 4 else 0, bias=False)
                    if s == 4 else nn.Conv2d(in_ch, d, kernel_size=s, stride=s, bias=False),
                    nn.BatchNorm2d(d), nn.ReLU(inplace=True),
                )
            )
        self.blocks = nn.ModuleList()
        for i in range(4):
            stage = nn.ModuleList([
                MiTBlock(dims[i], heads[i], sr[i]) for _ in range(depths[i])
            ])
            self.blocks.append(stage)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for pe, stage in zip(self.patch_embed, self.blocks):
            x = pe(x)
            for blk in stage:
                x = blk(x)
            feats.append(x)
        return feats


class MiTBlock(nn.Module):
    """Efficient self-attention + Mix-FFN (SegFormer block)."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        y = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(y, (h, w))
        x = x + self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False),
                nn.BatchNorm2d(dim),
            )
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        d_head = c // self.num_heads
        # q: (b, heads, n, d_head)   - tokens and channels must stay in separate axes
        q = self.q(x).reshape(b, self.num_heads, d_head, n).permute(0, 1, 3, 2)
        if self.sr_ratio > 1:
            xs = self.sr(x)
            xs = self.norm(xs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            kv = self.kv(xs)
        else:
            kv = self.kv(x)
        ns = kv.shape[-2] * kv.shape[-1]
        kv = kv.reshape(b, 2, self.num_heads, d_head, ns).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]                      # (b, heads, n_sr, d_head)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return self.proj(out)


class SegFormer(nn.Module):
    """SegFormer: MiT encoder + all-MLP head (MMSeg `segformer` config family).

    NOTE: MiT has no ImageNet weights in this environment (no timm), so this
    trains from scratch - use `SwinUNet` when pretrained features are wanted.
    """

    def __init__(self, variant: str = "b2", out_channels: int = 1,
                 pretrained: bool = False, embed_dim: int = 256):
        super().__init__()
        self.encoder = MiTEncoder(variant)
        self.head = SegFormerHead(self.encoder.dims, embed_dim, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats, x.shape[-2:])


class SwinUNet(nn.Module):
    """Swin-Transformer (ImageNet pretrained, torchvision) + U-Net decoder.

    Motivation: the in-house SegFormer trains from scratch (no MiT weights
    available), which is a real handicap at 1200 tiles.  torchvision ships
    ImageNet weights for Swin-T/V2-S, so this backbone starts from strong
    generic features while the decoder keeps full-resolution detail.

    Feature pyramid: torchvision `swin.features` alternates Siamese blocks and
    PatchMerging layers, so outputs at indices 1, 3, 5, 7 are the four stages
    with strides 4, 8, 16, 32.
    """

    STAGE_IDX = (1, 3, 5, 7)

    def __init__(self, variant: str = "swin_t", pretrained: bool = True,
                 out_channels: int = 1, decoder_ch: Sequence[int] = (128, 96, 64, 48),
                 head_stride: int = 4):
        super().__init__()
        import torchvision.models as tvm

        table = {
            "swin_t": (tvm.swin_t, lambda: tvm.Swin_T_Weights.IMAGENET1K_V1, [96, 192, 384, 768]),
            "swin_s": (tvm.swin_s, lambda: tvm.Swin_S_Weights.IMAGENET1K_V1, [96, 192, 384, 768]),
            "swin_v2_s": (tvm.swin_v2_s, lambda: tvm.Swin_V2_S_Weights.IMAGENET1K_V1,
                          [96, 192, 384, 768]),
            "swin_b": (tvm.swin_b, lambda: tvm.Swin_B_Weights.IMAGENET1K_V1,
                       [128, 256, 512, 1024]),
        }
        if variant not in table:
            raise ValueError(f"unsupported swin variant {variant}")
        ctor, weights_fn, widths = table[variant]
        net = ctor(weights=weights_fn() if pretrained else None)
        self.backbone = net.features
        self.widths = widths
        d0, d1, d2, d3 = decoder_ch
        # stages are channels-last (N,H,W,C); project to dense maps first
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(w, d, 1, bias=False), nn.BatchNorm2d(d),
                          nn.ReLU(inplace=True))
            for w, d in zip(widths, (d3, d3, d3, d3))
        ])
        self.dec3 = UpBlock(d3, d3, d2, attn=True)
        self.dec2 = UpBlock(d2, d3, d1, attn=True)
        self.dec1 = UpBlock(d1, d3, d0, attn=True)
        self.head = nn.Sequential(conv_bn_act(d0, 64), nn.Conv2d(64, out_channels, 1))
        # Optional finer head: one extra 2x upsample + a small conv at stride 2.
        # Rationale: the resolution ladder (1.0x -> 0.6514, 1.25x -> 0.6572,
        # 1.5x -> 0.6620) shows the *effective output stride* is the bottleneck, so
        # halving the head stride is a cheaper way to the same effect.
        self.head_stride = int(head_stride)
        if self.head_stride == 2:
            self.head2 = nn.Sequential(conv_bn_act(d0, 48), nn.Conv2d(48, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = []
        y = x
        for i, layer in enumerate(self.backbone):
            y = layer(y)
            if i in self.STAGE_IDX:
                feats.append(y)          # (N, H', W', C)
        maps = [F.interpolate(p(f.permute(0, 3, 1, 2).contiguous()),
                              size=(h // 4, w // 4), mode="bilinear", align_corners=False)
                for p, f in zip(self.proj, feats)]
        d = self.dec3(maps[3], maps[2])
        d = self.dec2(d, maps[1])
        d = self.dec1(d, maps[0])
        if self.head_stride == 2:
            d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
            out = self.head2(d)
        else:
            out = self.head(d)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------- ConvNeXtUNet


class ConvNeXtUNet(nn.Module):
    """ConvNeXt (ImageNet pretrained, torchvision) + U-Net decoder at stride 4.

    Mirrors `SwinUNet`'s winning layout: every stage's feature map is projected to a
    common width and resampled to stride 4, then fused by three gated up-blocks and a
    small head; the logits are upsampled to the exact input size.

    Why a second transformer-family member: the ensemble already has two Swin-S
    members (r~0.95 correlated). ConvNeXt is a *convolutional* ImageNet backbone with
    a modern (7x7 depthwise + inverted bottleneck) design, so its errors should be
    less correlated with Swin's than another Swin seed - the cheapest remaining way
    to raise ensemble diversity at the same quality level.
    """

    STAGE_IDX = (1, 3, 5, 7)   # /4, /8, /16, /32 stage outputs of torchvision convnext.features

    def __init__(self, variant: str = "convnext_t", pretrained: bool = True,
                 out_channels: int = 1, decoder_ch: Sequence[int] = (128, 96, 64),
                 proj_ch: int = 64, head_ch: int = 64):
        super().__init__()
        import torchvision.models as tvm

        table = {
            "convnext_t": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                           [96, 192, 384, 768]),
            "convnext_s": (tvm.convnext_small, tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1,
                           [96, 192, 384, 768]),
            "convnext_b": (tvm.convnext_base, tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1,
                           [128, 256, 512, 1024]),
        }
        if variant not in table:
            raise ValueError(f"unsupported convnext variant {variant}")
        ctor, weights_fn, widths = table[variant]
        w = weights_fn() if callable(weights_fn) else weights_fn
        net = ctor(weights=w if pretrained else None)
        self.backbone = net.features
        self.widths = widths
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(w, proj_ch, 1, bias=False), nn.BatchNorm2d(proj_ch),
                          nn.ReLU(inplace=True))
            for w in widths
        ])
        d0, d1, d2 = decoder_ch
        self.dec3 = UpBlock(proj_ch, proj_ch, d0, attn=True)
        self.dec2 = UpBlock(d0, proj_ch, d1, attn=True)
        self.dec1 = UpBlock(d1, proj_ch, d2, attn=True)
        self.head = nn.Sequential(conv_bn_act(d2, head_ch), nn.Conv2d(head_ch, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = []
        y = x
        for i, layer in enumerate(self.backbone):
            y = layer(y)
            if i in self.STAGE_IDX:
                feats.append(y)
        maps = [F.interpolate(p(f), size=(h // 4, w // 4), mode="bilinear",
                              align_corners=False)
                for p, f in zip(self.proj, feats)]
        d = self.dec3(maps[3], maps[2])
        d = self.dec2(d, maps[1])
        d = self.dec1(d, maps[0])
        out = self.head(d)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


# -------------------------------------------------------------------- losses


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + soft Dice (MMSeg `DiceLoss` with sigmoid)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return self.bce_weight * bce + self.dice_weight * dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., MMSeg `FocalLoss` with sigmoid)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == "mean" else loss.sum()


class TverskyLoss(nn.Module):
    """Tversky loss (Salehi et al. 2017); beta>alpha favours recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        tp = (p * targets).sum(dim=dims)
        fp = (p * (1 - targets)).sum(dim=dims)
        fn = ((1 - p) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


class SoftIoULoss(nn.Module):
    """Differentiable (Jaccard) IoU loss - directly optimises the metric."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (p * targets).sum(dim=dims)
        union = p.sum(dim=dims) + targets.sum(dim=dims) - inter
        return (1.0 - (inter + self.smooth) / (union + self.smooth)).mean()


class LovaszHingeLoss(nn.Module):
    """Lovasz-Softmax hinge variant (Berman et al. CVPR 2018).

    Implemented from the paper's algorithm; the per-image gradient/error
    computation is decoupled from the loss so it also works for binary masks.
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
        return jaccard

    def _flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # accept (N,1,H,W) or (N,H,W): the algorithm is defined per-pixel-plane
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        if self.per_image:
            losses = [self._flat(l.reshape(-1), t.reshape(-1)) for l, t in zip(logits, targets)]
            return torch.stack(losses).mean()
        return self._flat(logits.reshape(-1), targets.reshape(-1))


class CrackLoss(nn.Module):
    """Composite objective: BCE + Dice + Tversky (+ optional soft-IoU).

    Weights are configurable so the mix can be tuned per architecture.
    """

    def __init__(self, bce: float = 0.3, dice: float = 0.4, tversky: float = 0.3,
                 iou: float = 0.0, tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, pos_weight: float = 0.0):
        super().__init__()
        self.w = dict(bce=bce, dice=dice, tversky=tversky, iou=iou)
        self.bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight > 0 else None)
        self.dice_fn = BCEDiceLoss(0.0, 1.0)
        self.tversky_fn = TverskyLoss(tversky_alpha, tversky_beta)
        self.iou_fn = SoftIoULoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.bce_fn.pos_weight is not None:
            self.bce_fn.pos_weight = self.bce_fn.pos_weight.to(logits.device)
        loss = self.w["bce"] * self.bce_fn(logits, targets)
        if self.w["dice"]:
            loss = loss + self.w["dice"] * self.dice_fn(logits, targets)
        if self.w["tversky"]:
            loss = loss + self.w["tversky"] * self.tversky_fn(logits, targets)
        if self.w["iou"]:
            loss = loss + self.w["iou"] * self.iou_fn(logits, targets)
        return loss


# ------------------------------------------------------------------- ResUNet


class ResUNet(nn.Module):
    """ResNet encoder + lightweight stride-1 decoder (ResUNet family).

    Motivation: the leaderboard top-3 are ResUNet variants, and our error
    decomposition says the remaining gap is a 1 px boundary problem - i.e. the
    decoder has to end at stride 1.  A naive stride-1 decoder (dense 3x3 convs
    at 672x378) is what made `unetd` cost 6 min/epoch; here every high-resolution
    stage is depthwise-separable and the last two strides are plain bilinear +
    one separable 3x3, so the /1 work is ~0.5 GMAC instead of ~5 GMAC.
    """

    def __init__(self, encoder: str = "resnet34", pretrained: bool = True,
                 out_channels: int = 1, aspp: bool = True, attn: bool = True,
                 mid_ch: Sequence[int] = (192, 128, 96), top_ch: int = 48,
                 head_ch: int = 32, hi_res_stem: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained, replace_stem=hi_res_stem)
        self.hi_res_stem = hi_res_stem
        w = self.encoder.widths            # [stem, c1, c2, c3, c4]
        self.aspp = ASPP(w[4], 256) if aspp else None
        m0, m1, m2 = mid_ch
        self.dec4 = UpBlock(256 if aspp else w[4], w[3], m0, attn, style="ds")
        self.dec3 = UpBlock(m0, w[2], m1, attn, style="ds")
        self.dec2 = UpBlock(m1, w[1], m2, attn, style="ds")     # -> /4
        self.dec1 = UpBlock(m2, self.encoder.stem_out_ch, top_ch, attn,
                            style="ds")                          # -> /2
        self.head = nn.Sequential(
            ds_conv_bn_act(top_ch, head_ch),                     # /1, cheap
            nn.Conv2d(head_ch, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        c0, c1, c2, c3, c4 = self.encoder(x)      # /2, /4, /8, /16, /32
        d = self.aspp(c4) if self.aspp is not None else c4
        d = self.dec4(d, c3)
        d = self.dec3(d, c2)
        d = self.dec2(d, c1)
        d = self.dec1(d, c0)                      # /2
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(d)


# ----------------------------------------------------------------- factories

MODEL_TABLE = (
    # main architecture: geodesic stem + SCSE skips + ASPP + GIT-style gated attention
    "unet_r18", "unet_r34", "unet_r50",
    # ablations
    "unetx_r34",      # no gated attention at all
    "unetg_r34",      # gated attention, channel gate only
    "unets_r34",      # gated attention, spatial gate only
    "unetp_r34",      # plain stride-2 torchvision stem, no ASPP
    # speed/accuracy trade-offs (see bench.py)
    "unetd_r34",      # depthwise-separable dilated decoder (fast, MMSeg-style)
    "unetd_r50",
    # other families
    "deeplabv3plus_r34", "deeplabv3plus_se_r34",
    "resunet_r34", "resunet_r50",   # stride-1 residual U-Net (cheap high-res decoder)
    "resuneth_r34", "resuneth_r50",  # same, but with the geodesic stride-1 stem
    "segformer_b0", "segformer_b2", "segformer_b3",
    "swin_t", "swin_s", "swin_b",     # ImageNet-pretrained transformer backbones
    "convnext_t", "convnext_s", "convnext_b",   # ImageNet-pretrained ConvNeXt family
)

_ENCODER_ALIAS = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50",
                  "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50"}


def build_model(name: str = "unet_r34", pretrained: bool = True, **kw) -> nn.Module:
    """Factory used by train.py; `name` encodes backbone + head choice."""
    parts = name.split("_")
    variant = parts[0]

    if variant.startswith("swin"):
        head_stride = 4
        tail = parts[-1]
        if tail == "s2":            # e.g. "swin_s_s2" -> stride-2 head
            head_stride = 2
            tail = parts[-2] if len(parts) > 2 else "s"
        swin_variant = "swin_" + tail if len(parts) > 1 else "swin_t"
        return SwinUNet(variant=swin_variant, pretrained=pretrained,
                        head_stride=head_stride, **kw)

    if variant == "convnext":
        return ConvNeXtUNet(variant="convnext_" + parts[-1], pretrained=pretrained, **kw)

    if variant in ("unet", "unetx", "unetg", "unets", "unetp", "unetd"):
        enc = _ENCODER_ALIAS[parts[-1]]
        opts: Dict = dict(kw)
        if variant == "unetx":
            opts.update(attn=False)
        elif variant == "unetg":
            opts.update(gat_mode="channel")
        elif variant == "unets":
            opts.update(gat_mode="spatial")
        elif variant == "unetp":
            opts.update(plain_stem=True, aspp=False, attn=False)
        elif variant == "unetd":
            opts.update(style="ds", dilate_high_res=True)
        return CrackUNet(encoder=enc, pretrained=pretrained, **opts)

    if variant in ("resunet", "resuneth"):
        return ResUNet(encoder=_ENCODER_ALIAS[parts[-1]], pretrained=pretrained,
                       hi_res_stem=(variant == "resuneth"), **kw)

    if variant == "deeplabv3plus":
        if parts[1] == "se":
            return DeepLabV3Plus(encoder=_ENCODER_ALIAS[parts[-1]], pretrained=pretrained,
                                 se=True, **kw)
        return DeepLabV3Plus(encoder=_ENCODER_ALIAS[parts[1]], pretrained=pretrained, **kw)

    if variant == "segformer":
        return SegFormer(variant=parts[-1], **kw)

    raise ValueError(f"unknown model {name!r}; known: {MODEL_TABLE}")


def build_loss(name: str = "bce_dice", **kw) -> nn.Module:
    if name == "bce_dice":
        return BCEDiceLoss(kw.get("bce", 0.5), kw.get("dice", 0.5))
    if name == "crack":
        return CrackLoss(**{k: v for k, v in kw.items() if k in
                            ("bce", "dice", "tversky", "iou", "tversky_alpha",
                             "tversky_beta", "pos_weight")})
    if name == "tversky":
        return TverskyLoss(**{k: v for k, v in kw.items() if k in ("alpha", "beta")})
    if name == "focal":
        return FocalLoss(**{k: v for k, v in kw.items() if k in ("alpha", "gamma")})
    if name == "lovasz":
        return LovaszHingeLoss()
    raise ValueError(f"unknown loss {name!r}")
