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

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class DistillProjectorHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(in_dim)
        if in_dim == out_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x.flatten(2)

class DistillProjector(nn.Module):
    def __init__(
        self,
        fpn_dim=256,
        proj_dim=256,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.level1 = DistillProjectorHead(fpn_dim, proj_dim, norm_layer,)
        self.level2 = DistillProjectorHead(fpn_dim, proj_dim // 4, norm_layer,)
        self.level3 = DistillProjectorHead(fpn_dim, proj_dim // 16, norm_layer,)

    def forward(self, feats):
        x1 = self.level1(feats[0])   # [B, 256]
        x2 = self.level2(feats[1])   # [B, 256]
        x3 = self.level3(feats[2])   # [B, 256]

        # iTPN integral pyramid fusion
        x = x1 + x2 + x3

        return x

class FPNProjectorHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = x.mean(dim=1)       # [B, h, w, C]
        x = self.proj(x)
        return x.flatten(1)

class DeCURFPNProjector(nn.Module):
    def __init__(
        self,
        fpn_dim=256,
        proj_dim=8192,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.level1 = FPNProjectorHead(fpn_dim, proj_dim, norm_layer,)
        self.level2 = FPNProjectorHead(fpn_dim, proj_dim // 4, norm_layer,)
        self.level3 = FPNProjectorHead(fpn_dim, proj_dim // 16, norm_layer,)

        self.bn1 = nn.BatchNorm1d(proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim,bias=False)
        self.bn2 = nn.BatchNorm1d(proj_dim)
        self.fc3 = nn.Linear(proj_dim, proj_dim,bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        x1 = self.level1(feats[0])   # [B, 8192]
        x2 = self.level2(feats[1])   # [B, 8192]
        x3 = self.level3(feats[2])   # [B, 8192]

        # iTPN integral pyramid fusion
        x = x1 + x2 + x3

        # original DeCUR projector tail
        x = self.relu(self.bn1(x))

        x = self.fc2(x)
        x = self.relu(self.bn2(x))

        x = self.fc3(x)

        return x

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

        if self.dino:
            dpr_cls = list(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))

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

        # ------------------multi-scale projector (adapt for iTPN)-------------------------
        self.projector_rgb = DeCURFPNProjector(fpn_dim=fpn_dim, proj_dim=8192, norm_layer=norm_layer)
        self.projector_sar = DeCURFPNProjector(fpn_dim=fpn_dim, proj_dim=8192, norm_layer=norm_layer)
        self.bn = nn.BatchNorm1d(8192, affine=False)

        if self.dino:
            self.distil_projector = DistillProjector(fpn_dim=fpn_dim, proj_dim=fpn_dim, norm_layer=norm_layer)

        if self.dino:
            self.lm_head = nn.Linear(fpn_dim, 1024)
            self.lm_cls_head = nn.Linear(embed_dim, 1024)
            self.norm = norm_layer(fpn_dim)
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

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.absolute_pos_embed.shape[-1], Hp, cls_token=False)
        self.absolute_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.apply(self._init_weights)
        self._init_distributed()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_distributed(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.distributed = True
        else:
            self.world_size = 1
            self.rank = 0
            self.distributed = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

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
                if self.dino:
                    aux_out = x_rgb
                else:
                    aux_out = None

        if self.dino:
            for blk in self.cls_pt_layers:
                aux_out = blk(aux_out, rpe_index)


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

    def forward_encoder(self, x_rgb, x_sar):

        x_rgb, x_sar, aux_out = self.forward_features(x_rgb, x_sar)

        return x_rgb, x_sar, aux_out

    def bt_loss_cross(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0] * self.world_size)

        if self.distributed:
            torch.distributed.all_reduce(c)

        dim_c = 7168
        c_c = c[:dim_c, :dim_c]
        c_u = c[dim_c:, dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()

        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()

        loss_c = on_diag_c + 0.0051 * off_diag_c
        loss_u = on_diag_u + 0.0051 * off_diag_u

        return loss_c, on_diag_c, off_diag_c, loss_u, on_diag_u, off_diag_u

    def bt_loss_single(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0] * self.world_size)

        if self.distributed:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.0051 * off_diag
        return loss, on_diag, off_diag

    def forward(self, imgs, tea_feat=None):
        inputs = imgs['normalized']
        x_rgb_1 = inputs[:, :3, :, :].cuda(non_blocking=True)
        x_rgb_2 = inputs[:, 3:6, :, :].cuda(non_blocking=True)
        x_sar_1 = inputs[:, 6:9, :, :].cuda(non_blocking=True)
        x_sar_2 = inputs[:, 9:, :, :].cuda(non_blocking=True)

        x_rgb = torch.cat([x_rgb_1, x_rgb_2], dim=0)
        x_sar = torch.cat([x_sar_1, x_sar_2], dim=0)

        latent_rgb, latent_sar, aux_x = self.forward_encoder(x_rgb, x_sar)

        z_rgb = self.projector_rgb(latent_rgb)
        z_sar = self.projector_sar(latent_sar)
        B = z_rgb.shape[0] // 2

        if self.dino:
            x_rgb_distill = self.distil_projector(latent_rgb)[:B]
            aux_x = aux_x[:B]
            dis_out = self.norm(x_rgb_distill)
            dis_out = self.lm_head(dis_out)
            dis_out_aux = self.norm_cls(aux_x)
            dis_out_aux = self.lm_cls_head(dis_out_aux)
            distill_loss_main = nn.SmoothL1Loss()(input=dis_out, target=tea_feat)
            distill_loss_aux = nn.SmoothL1Loss()(input=dis_out_aux, target=tea_feat)
            distill_loss = (distill_loss_main + distill_loss_aux)  # / 2
        else:
            distill_loss = 0.

        z1_1 = z_rgb[:B]
        z1_2 = z_rgb[B:]
        z2_1 = z_sar[:B]
        z2_2 = z_sar[B:]

        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1, z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1, z2_2)
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1, z2_1)
        loss12 = (loss12_c + loss12_u) / 2.0

        total_loss = (loss1 + loss2 + loss12) / 3

        return total_loss, distill_loss

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