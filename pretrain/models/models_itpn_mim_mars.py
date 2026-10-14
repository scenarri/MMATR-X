# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


from typing import Iterable
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from util.pos_embed import get_2d_sincos_pos_embed
import torch.utils.checkpoint as checkpoint
import math
import torch.nn.functional as F
from torch import distributed as dist


def cross_contrastive(rgb_feats, sar_feats,
                      temp_patch=0.1, temp_global=0.05, temp_hier=0.07,
                      alpha=0.5, beta=0.3, gamma=0.2):
    """
    多层次跨模态对比损失函数
    Args:
        rgb_feats: RGB特征 [B, N, D]
        sar_feats: SAR特征 [B, N, D]
        temp_*: 各层级的温度系数
        alpha, beta, gamma: 损失权重
    Returns:
        contrastive_loss: 综合对比损失
    """

    # ==================== 特征预处理 ====================
    # 移除CLS token并归一化 (假设输入已处理)
    rgb_patches = F.normalize(rgb_feats, p=2, dim=-1)
    sar_patches = F.normalize(sar_feats, p=2, dim=-1)
    B, N, D = rgb_patches.shape

    # ==================== 全局特征提取 ====================
    rgb_global = rgb_patches.mean(dim=1)  # [B, D]
    sar_global = sar_patches.mean(dim=1)

    # ==================== 核心对比计算 ====================
    # 1. Patch-to-Patch对比
    patch_sim = torch.einsum('bnd,jmd->bnjm', rgb_patches, sar_patches) / temp_patch
    patch_sim = patch_sim.reshape(B * N, B * N)  # [BN, BN]
    # 2. Image-to-Image对比
    global_sim = torch.mm(rgb_global, sar_global.T) / temp_global  # [B, B]

    # 3. 双向Hierarchical对比
    rgb_global_expanded = rgb_global.unsqueeze(1).repeat(1, N, 1)
    sar_global_expanded = sar_global.unsqueeze(1).repeat(1, N, 1)

    crossrgb_sim = torch.einsum('bnd,jmd->bnjm', rgb_patches, sar_global_expanded) / temp_hier
    crossrgb_sim = crossrgb_sim.reshape(B * N, B * N)  # [BN, BN]
    crosssar_sim = torch.einsum('bnd,jmd->bnjm', sar_patches, rgb_global_expanded) / temp_hier
    crosssar_sim = crosssar_sim.reshape(B * N, B * N)  # [BN, BN]

    # ==================== 损失计算 ====================
    device = rgb_feats.device

    # 1. Patch-level损失
    patch_labels = torch.arange(B * N, device=device)
    patch_loss = (F.cross_entropy(patch_sim, patch_labels) +
                  F.cross_entropy(patch_sim.T, patch_labels)) / 2

    # 2. Image-level损失
    img_labels = torch.arange(B, device=device)
    img_loss = (F.cross_entropy(global_sim, img_labels) +
                F.cross_entropy(global_sim.T, img_labels)) / 2

    # 3. Hierarchical损失
    hier_labels = torch.arange(B, device=device).repeat_interleave(N)  # [BN]

    hier_loss = (F.cross_entropy(crossrgb_sim, hier_labels) +
                 F.cross_entropy(crosssar_sim, hier_labels)) / 2

    # ==================== 加权融合 ====================
    total_loss = (alpha * patch_loss +
                  beta * img_loss +
                  gamma * hier_loss)

    return total_loss


class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                        4)  # 3, 4, 8, 49, 64
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 4, 8, 49, 64
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MetaAttention(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=8, mlp_ratio=4.0, drop=0.0, aa_order=('modality', 'global')):
        super().__init__()
        self.aa_block_num = num_layers
        self.aa_order = aa_order

        self.modality_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                DropPath(drop) if drop > 0 else nn.Identity(),
                Mlp(dim, int(dim * mlp_ratio), drop=drop),
            ]) for _ in range(num_layers)
        ])
        self.global_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                DropPath(drop) if drop > 0 else nn.Identity(),
                Mlp(dim, int(dim * mlp_ratio), drop=drop),
            ]) for _ in range(num_layers)
        ])

    def forward(self, rgb_feat, sar_feat):
        B, S, C = rgb_feat.shape
        tokens = torch.cat([rgb_feat, sar_feat], dim=1)  # [B, 2S, C]

        modality_idx = 0
        global_idx = 0

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "modality":
                    ln, attn, dp, mlp = self.modality_blocks[modality_idx]
                    t = ln(tokens)
                    rgb_tok = t[:, :S]
                    sar_tok = t[:, S:]
                    out_rgb, _ = attn(rgb_tok, rgb_tok, rgb_tok)
                    out_sar, _ = attn(sar_tok, sar_tok, sar_tok)
                    merged = torch.cat([out_rgb, out_sar], dim=1)
                    tokens = tokens + dp(mlp(merged))
                    modality_idx += 1
                elif attn_type == "global":
                    ln, attn, dp, mlp = self.global_blocks[global_idx]
                    t = ln(tokens)
                    out, _ = attn(t, t, t)
                    tokens = tokens + dp(mlp(out))
                    global_idx += 1
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

        rgb_out = tokens[:, :S, :]
        sar_out = tokens[:, S:, :]

        return rgb_out, sar_out

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchSplit(nn.Module):
    def __init__(self, dim, fpn_dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, fpn_dim * 4, bias=False)
        self.fpn_dim = fpn_dim

    def forward(self, x):
        B, N, H, W, C = x.shape
        x = self.norm(x)
        x = self.reduction(x)
        x = x.reshape(
            B, N, H, W, 2, 2, self.fpn_dim
        ).permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, N, 2 * H, 2 * W, self.fpn_dim
        )
        return x


class iTPNMaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, fpn_dim=256, fpn_depth=1,
                 embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=True,
                 num_outs=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, modality='Both',
                 norm_pix_loss=True, mask_ratio=0.75, dino=False, **kwargs):
        super().__init__()
        assert num_outs in [1, 2, 3]
        self.num_classes = num_classes
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_outs = num_outs
        self.num_main_blocks = depth
        self.fpn_dim = fpn_dim
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.modality = modality
        print(f'MAE initialzied with {self.modality} modality')

        self.norm_pix_loss = norm_pix_loss
        print(f'norm_pix_loss: {self.norm_pix_loss}')

        self.mask_ratio = mask_ratio
        self.dino = dino
        print(f'distilled from dino: {self.dino}')

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        self.patch_embed_rgb = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_sar = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed_sar.num_patches
        Hp, Wp = self.patch_embed_sar.patches_resolution

        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))

        # if self.dino:
        dpr_cls = list(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))

        self.blocks_rgb = nn.ModuleList()

        self.blocks_rgb.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks_rgb.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks_rgb.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks_rgb.append(PatchMerge(mlvl_dims['8'], norm_layer))

        self.blocks_rgb.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(depth)]
        )

        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))

        self.blocks_sar = nn.ModuleList()

        self.blocks_sar.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks_sar.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks_sar.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks_sar.append(PatchMerge(mlvl_dims['8'], norm_layer))

        self.blocks_sar.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(depth)]
        )

        ########################### FPN PART ###########################
        if self.num_outs > 1:
            self.align_dim_16tofpn_rgb = nn.Linear(embed_dim, fpn_dim) if embed_dim != fpn_dim else None
            self.fpn_modules_rgb = nn.ModuleList()
            self.fpn_modules_rgb.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                ))

            self.align_dim_16to8_rgb = nn.Linear(mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8_rgb = PatchSplit(mlvl_dims['16'], fpn_dim, norm_layer)
            self.block_16to8_rgb = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules_rgb.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ))

        if self.num_outs > 2:
            self.align_dim_8to4_rgb = nn.Linear(mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4_rgb = PatchSplit(fpn_dim, fpn_dim, norm_layer)
            self.block_8to4_rgb = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules_rgb.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                )
            )

        if self.num_outs > 1:
            self.align_dim_16tofpn_sar = nn.Linear(embed_dim, fpn_dim) if embed_dim != fpn_dim else None
            self.fpn_modules_sar = nn.ModuleList()
            self.fpn_modules_sar.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                ))

            self.align_dim_16to8_sar = nn.Linear(mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8_sar = PatchSplit(mlvl_dims['16'], fpn_dim, norm_layer)
            self.block_16to8_sar = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules_sar.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ))

        if self.num_outs > 2:
            self.align_dim_8to4_sar = nn.Linear(mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4_sar = PatchSplit(fpn_dim, fpn_dim, norm_layer)
            self.block_8to4_sar = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules_sar.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                )
            )

        # --------------------------------------------------------------------------

        ########################### Meta-Attention Loss ###########################
        self.metaattn = MetaAttention(dim=decoder_embed_dim)


        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_size = patch_size

        self.decoder_embed_rgb = nn.ModuleList()
        self.decoder_embed_rgb.append(
            nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim, bias=True)))
        if self.num_outs >= 2:
            self.decoder_embed_rgb.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 4, bias=True)))
        if self.num_outs >= 3:
            self.decoder_embed_rgb.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 16, bias=True)))

        self.decoder_embed_sar = nn.ModuleList()
        self.decoder_embed_sar.append(
            nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim, bias=True)))
        if self.num_outs >= 2:
            self.decoder_embed_sar.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 4, bias=True)))
        if self.num_outs >= 3:
            self.decoder_embed_sar.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 16, bias=True)))

        self.mask_token_rgb = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token_rgb, std=.02)

        self.mask_token_sar = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token_sar, std=.02)

        self.decoder_blocks_rgb = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                rpe=False, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)])

        self.decoder_blocks_sar = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                rpe=False, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)])

        self.decoder_norm_rgb = norm_layer(decoder_embed_dim)
        self.decoder_pred_rgb = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.decoder_norm_sar = norm_layer(decoder_embed_dim)
        self.decoder_pred_sar = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        if self.dino:
            pred_head_dim = 1024
        else:
            pred_head_dim = 512
        print(f'pead dim = {pred_head_dim}')
        self.lm_head = nn.Linear(decoder_embed_dim, pred_head_dim)
        self.lm_cls_head = nn.Linear(embed_dim, pred_head_dim)
        self.norm = norm_layer(decoder_embed_dim)
        self.norm_cls = norm_layer(embed_dim)
        trunc_normal_(self.lm_head.weight, std=.02)
        trunc_normal_(self.lm_cls_head.weight, std=.02)
        self.cls_pt_layers = nn.ModuleList(
            [
                BlockWithRPE(input_size=Hp, dim=mlvl_dims['16'], num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop_path=dpr_cls[2 * self.mlp_depth + i], rpe=rpe,
                             norm_layer=norm_layer)
                for i in range(int(self.depth * 0.75), int(self.depth * 0.75) + 2)
            ]
        )

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.absolute_pos_embed.shape[-1], Hp, cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], Hp, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.absolute_pos_embed.size(1)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=self.absolute_pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.absolute_pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x_rgb, x_sar, ids_keep=None, mask=None):

        B, C, H, W = x_rgb.shape

        x_rgb = self.patch_embed_rgb(x_rgb)
        x_sar = self.patch_embed_sar(x_sar)
        Hp, Wp = H // self.patch_embed_rgb.patch_size[0], W // self.patch_embed_rgb.patch_size[1]

        if ids_keep is not None:
            x_rgb = torch.gather(x_rgb, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x_rgb.shape[2:]))
            x_sar = torch.gather(x_sar, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x_sar.shape[2:]))

        features_rgb = []
        for blk in self.blocks_rgb[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                features_rgb.append(x_rgb)
            x_rgb = checkpoint.checkpoint(blk, x_rgb) if self.use_checkpoint else blk(x_rgb)

        x_rgb = x_rgb[..., 0, 0, :]
        if self.ape:
            pos_embed = self.interpolate_pos_encoding(x_rgb, Hp, Wp)
            if ids_keep is not None:
                pos_embed = torch.gather(pos_embed.expand(B, -1, -1), dim=1, index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),)
            x_rgb += pos_embed
        x_rgb = self.pos_drop(x_rgb)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for j, blk in enumerate(self.blocks_rgb[-self.num_main_blocks:]):
            x_rgb = checkpoint.checkpoint(blk, x_rgb, rpe_index, mask) if self.use_checkpoint else blk(x_rgb, rpe_index, mask)
            if j == int(0.75 * self.depth):
                # if self.dino:
                aux_out = x_rgb
            # else:
            #         aux_out = None

        features_sar = []
        for blk in self.blocks_sar[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                features_sar.append(x_sar)
            x_sar = checkpoint.checkpoint(blk, x_sar) if self.use_checkpoint else blk(x_sar)

        x_sar = x_sar[..., 0, 0, :]
        if self.ape:
            pos_embed = self.interpolate_pos_encoding(x_sar, Hp, Wp)
            if ids_keep is not None:
                pos_embed = torch.gather(pos_embed.expand(B, -1, -1), dim=1, index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),)
            x_sar += pos_embed
        x_sar = self.pos_drop(x_sar)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for j, blk in enumerate(self.blocks_sar[-self.num_main_blocks:]):
            x_sar = checkpoint.checkpoint(blk, x_sar, rpe_index, mask) if self.use_checkpoint else blk(x_sar, rpe_index, mask)

        if self.num_outs == 1:
            return x_rgb, x_sar

        ##########################  FPN forward  ########################

        x_rgb = x_rgb[..., None, None, :]
        outs_rgb = [x_rgb] if self.align_dim_16tofpn_rgb is None else [self.align_dim_16tofpn_rgb(x_rgb)]
        if self.num_outs >= 2:
            x_rgb = self.block_16to8_rgb(self.split_16to8_rgb(x_rgb) + self.align_dim_16to8_rgb(features_rgb[1]))
            outs_rgb.append(x_rgb)
        if self.num_outs >= 3:
            x_rgb = self.block_8to4_rgb(self.split_8to4_rgb(x_rgb) + self.align_dim_8to4_rgb(features_rgb[0]))
            outs_rgb.append(x_rgb)
        if rpe_index is None and self.num_outs > 3:
            outs_rgb = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                    B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
                for out in outs_rgb]

            if self.num_outs >= 4:
                outs_rgb.insert(0, F.avg_pool2d(outs_rgb[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs_rgb.insert(0, F.avg_pool2d(outs_rgb[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs_rgb):
            out = self.fpn_modules_rgb[i](out)
            outs_rgb[i] = out

        # if self.dino:
        for blk in self.cls_pt_layers:
            aux_out = blk(aux_out, rpe_index)

        x_sar = x_sar[..., None, None, :]
        outs_sar = [x_sar] if self.align_dim_16tofpn_sar is None else [self.align_dim_16tofpn_sar(x_sar)]
        if self.num_outs >= 2:
            x_sar = self.block_16to8_sar(self.split_16to8_sar(x_sar) + self.align_dim_16to8_sar(features_sar[1]))
            outs_sar.append(x_sar)
        if self.num_outs >= 3:
            x_sar = self.block_8to4_sar(self.split_8to4_sar(x_sar) + self.align_dim_8to4_sar(features_sar[0]))
            outs_sar.append(x_sar)
        if rpe_index is None and self.num_outs > 3:
            outs_sar = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                    B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
                for out in outs_sar]

            if self.num_outs >= 4:
                outs_sar.insert(0, F.avg_pool2d(outs_sar[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs_sar.insert(0, F.avg_pool2d(outs_sar[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs_sar):
            out = self.fpn_modules_sar[i](out)
            outs_sar[i] = out

        return outs_rgb, outs_sar, aux_out

    def forward_encoder(self, x_rgb, x_sar, mask_ratio):

        ids_keep, ids_restore, mask = self.masking_id(x_rgb.size(0), mask_ratio)
        # if self.modality == 'Both':
        #     ids_keep = torch.cat([ids_keep, ids_keep], dim=0)
        #     ids_restore = torch.cat([ids_restore, ids_restore], dim=0)
        #     mask = torch.cat([mask, mask], dim=0)

        x_rgb, x_sar, aux_x = self.forward_features(x_rgb, x_sar, ids_keep=ids_keep)

        return x_rgb, x_sar, aux_x, mask, ids_restore, ids_keep

    def forward_decoder(self, x_rgb, x_sar, ids_restore):  # latent, ids_restore
        B, N, _ = x_rgb.shape

        # tokens for self-modality reconstruction (RGB)
        mask_tokens = self.mask_token_rgb.repeat(x_rgb.shape[0], ids_restore.shape[1] - x_rgb.shape[1], 1)
        x_rgb = torch.cat([x_rgb, mask_tokens], dim=1)
        x_rgb = torch.gather(x_rgb, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_rgb.shape[2]))
        x_self_rgb = x_rgb + self.decoder_pos_embed

        # tokens for self-modality reconstruction (SAR)
        mask_tokens = self.mask_token_sar.repeat(x_sar.shape[0], ids_restore.shape[1] - x_sar.shape[1], 1)
        x_sar = torch.cat([x_sar, mask_tokens], dim=1)
        x_sar = torch.gather(x_sar, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_sar.shape[2]))
        x_self_sar = x_sar + self.decoder_pos_embed

        for i, blk in enumerate(self.decoder_blocks_rgb):
            x_self_rgb = checkpoint.checkpoint(blk, x_self_rgb) if self.use_checkpoint else blk(x_self_rgb)
        x_self_rgb = self.decoder_norm_rgb(x_self_rgb)
        x_self_rgb = self.decoder_pred_rgb(x_self_rgb)

        for i, blk in enumerate(self.decoder_blocks_sar):
            x_self_sar = checkpoint.checkpoint(blk, x_self_sar) if self.use_checkpoint else blk(x_self_sar)
        x_self_sar = self.decoder_norm_sar(x_self_sar)
        x_self_sar = self.decoder_pred_sar(x_self_sar)

        self_dec = torch.cat([x_self_rgb, x_self_sar], dim=0)

        return self_dec

    def forward_loss(self, x_rgb, x_sar, pred, mask):

        mask = torch.cat([mask, mask], dim=0)
        num_preds = mask.sum()
        target = torch.cat([self.patchify(x_rgb), self.patchify(x_sar)], dim=0)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds

        return loss

    def forward(self, imgs, mask_ratio=0.75, tea_feat=None):
        inputs = imgs['normalized']
        x_rgb = inputs[:, :3, :, :].cuda(non_blocking=True)
        x_sar = inputs[:, 3:, :, :].cuda(non_blocking=True)

        latent_rgb, latent_sar, aux_x, mask, ids_restore, ids_keep = self.forward_encoder(x_rgb, x_sar, mask_ratio)

        B, N, _, _, _ = latent_rgb[0].shape
        feats_rgb = []
        feats_sar = []
        for feat_rgb, feat_sar, layer_rgb, layer_sar in zip(latent_rgb, latent_sar, self.decoder_embed_rgb, self.decoder_embed_sar):
            feat_rgb_ = layer_rgb(feat_rgb).reshape(B, N, -1)
            feat_sar_ = layer_sar(feat_sar).reshape(B, N, -1)
            feats_rgb.append(feat_rgb_)
            feats_sar.append(feat_sar_)

        distill_rgb = []
        for x in feats_rgb:
            distill_rgb.append(x)
        x_rgb_distill = distill_rgb.pop(0)
        for i, feat in enumerate(distill_rgb):
            x_rgb_distill = x_rgb_distill + distill_rgb[i]

        distill_sar = []
        for x in feats_sar:
            distill_sar.append(x)
        x_sar_distill = distill_sar.pop(0)
        for i, feat in enumerate(distill_sar):
            x_sar_distill = x_sar_distill + distill_sar[i]

        x_rgb_distill_meta, x_sar_distill_meta = self.metaattn(x_rgb_distill, x_sar_distill)
        contrastive_loss = cross_contrastive(x_rgb_distill_meta, x_sar_distill_meta)

        self_pred = self.forward_decoder(x_rgb_distill_meta, x_sar_distill_meta, ids_restore)
        self_loss = self.forward_loss(x_rgb, x_sar, self_pred, mask)

        # if self.dino:
        dis_out = self.norm(x_rgb_distill_meta)
        # dis_out = self.norm(x_rgb_distill)
        dis_out = self.lm_head(dis_out)
        dis_out_aux = self.norm_cls(aux_x)
        dis_out_aux = self.lm_cls_head(dis_out_aux)
        tea_feat = torch.gather(tea_feat, dim=1, index=ids_keep[:, :, None].expand(-1, -1, tea_feat.shape[2]))
        distill_loss_main = nn.SmoothL1Loss()(input=dis_out, target=tea_feat)
        distill_loss_aux = nn.SmoothL1Loss()(input=dis_out_aux, target=tea_feat)
        distill_loss = (distill_loss_main + distill_loss_aux)  # / 2
        # else:
        #     distill_loss = 0.

        return self_loss, distill_loss, contrastive_loss, self_pred, mask

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        chs = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], chs, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * chs))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.decoder_patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        chs = x.shape[-1] // (h * w)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, chs))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], chs, h * p, h * p))
        return imgs


def itpn_tiny_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=384, mlp_depth=1, depth=10, num_heads=6, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def itpn_small_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=384, mlp_depth=3, depth=24, num_heads=6, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def itpn_base_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=512, mlp_depth=3, depth=24, num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def itpn_large_dec512d8b(**kwargs):
    model = iTPNMaskedAutoencoder(
        embed_dim=768, mlp_depth=2, depth=40, num_heads=12, bridge_mlp_ratio=3., mlp_ratio=4.,
        num_outs=3, decoder_embed_dim=512, decoder_num_heads=16,
        rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    model = itpn_base_dec512d8b(modality='Both',
                                cr_layers=8,
                                dec_share=True,
                                decoder_depth=8,
                                global_con=True,
                                conditioned=True,
                                norm_pix_loss=True)

    model.cuda()
    x = torch.randn(4, 6, 224, 224).cuda()
    x = {'normalized': x,
         'paired': torch.tensor([1, 1, 1, 1])}
    loss = model(x)
    print(loss)