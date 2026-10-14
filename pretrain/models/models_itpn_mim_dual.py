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
from features import DualFeat


def gather_features(features, world_size):
    if dist.is_available() and dist.is_initialized():
        if world_size == 1:
            return features

        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(features_list, features)
        all_features = torch.cat(features_list, dim=0)

        return all_features
    else:
        return features

class ContrastiveLoss(nn.Module):
    def __init__(self, in_dim=512, out_dim=256, conditioned=False, proj='none',
                 input_size=14, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        if conditioned:
            self.crossattn_rgb = BlockWithRPEMHCA(
                input_size, dim=in_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer,
                norm_layer=norm_layer, crossattn=True)

            self.crossattn_sar = BlockWithRPEMHCA(
                input_size, dim=in_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer,
                norm_layer=norm_layer, crossattn=True)

        if proj != 'none':
            if proj == 'linear':
                self.proj_rgb = nn.Linear(in_dim, out_dim)
                self.proj_sar = nn.Linear(in_dim, out_dim)
            elif proj == 'mlp':
                self.proj_rgb = nn.Sequential(nn.Linear(in_dim, 4 * in_dim), nn.GELU(), nn.Linear(4 * in_dim, out_dim))
                self.proj_sar = nn.Sequential(nn.Linear(in_dim, 4 * in_dim), nn.GELU(), nn.Linear(4 * in_dim, out_dim))
            else:
                raise NotImplementedError

        self.logit_scale = torch.log(torch.tensor(1.0 / 0.07))
        self.logit_scale.requires_grad = False

        self.conditioned = conditioned
        self.proj = proj

        self._init_distributed()

    def _init_distributed(self):
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.distributed = True
        else:
            self.world_size = 1
            self.rank = 0
            self.distributed = False

    def get_temperature(self):
        temperature = self.logit_scale.exp()
        return temperature

    def forward(self, x_rgb_in, x_sar_in):

        if self.conditioned:
            x_rgb = self.crossattn_rgb(x=x_rgb_in, context=x_sar_in)
            x_sar = self.crossattn_sar(x=x_sar_in, context=x_rgb_in)
            x_rgb = x_rgb.mean(dim=1)
            x_sar = x_sar.mean(dim=1)
        else:
            x_rgb = x_rgb_in.mean(dim=1)
            x_sar = x_sar_in.mean(dim=1)

        if self.proj != 'none':
            x_rgb = self.proj_rgb(x_rgb)
            x_sar = self.proj_sar(x_sar)

        x_rgb = F.normalize(x_rgb, p=2, dim=-1)
        x_sar = F.normalize(x_sar, p=2, dim=-1)

        if self.distributed:
            x_rgb_all = gather_features(x_rgb, self.world_size)  # (B_all, D_out)
            x_sar_all = gather_features(x_sar, self.world_size)  # (B_all, D_out)
        else:
            x_rgb_all = x_rgb
            x_sar_all = x_sar

        temperature = self.get_temperature()

        logits_per_rgb = temperature * x_rgb @ x_sar_all.t()
        logits_per_sar = temperature * x_sar @ x_rgb_all.t()

        num_logits = logits_per_rgb.shape[0]
        labels = torch.arange(num_logits, device=x_rgb.device, dtype=torch.long)
        labels = labels + num_logits * self.rank

        global_loss = (F.cross_entropy(logits_per_rgb, labels) + F.cross_entropy(logits_per_sar, labels)) / 2
        return global_loss


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
                 drop=0., attn_drop=0., drop_path=0., rpe=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 crossattn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.crossattn = crossattn

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        if crossattn:
            self.cattn = CrossAttention(
                input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, rpe=rpe, in_norm=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None, context=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        if self.crossattn and context is not None:
            x = x + self.drop_path(self.cattn(x, context, rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.,
                 in_norm=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.in_norm = in_norm

        if self.in_norm:
            self.n_q = norm_layer(dim)
            self.n_kv = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_kv):
        B, N, C = x_q.shape

        if self.in_norm:
            x_q = self.n_q(x_q)
            x_kv = self.n_kv(x_kv)

        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockWithRPEMHCA(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 crossattn=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.crossattn = crossattn
        self.cattn = CrossAttention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, in_norm=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, context):
        x = x + self.drop_path(self.cattn(x, context))
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
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 num_outs=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, modality='RGB',
                 norm_pix_loss=True, mask_ratio=0.75, cr_layers=0,
                 dino=False, contrastive=False, conditioned=False, proj='none',
                 target='pixel', reduction='none',
                 **kwargs):
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
        self.proj = proj
        print(f'MAE initialzied with {self.modality} modality')

        self.norm_pix_loss = norm_pix_loss
        print(f'norm_pix_loss: {self.norm_pix_loss}')

        self.mask_ratio = mask_ratio
        self.cr_layers = cr_layers
        self.target = target
        self.reduction = reduction
        print(
            f'Cross modality reconstruction by {self.cr_layers} layers, target: {self.target} with {self.reduction}')

        self.dino = dino
        print(f'distilled from dino: {self.dino}')

        self.contrastive = contrastive
        self.conditioned = conditioned
        print(f'contrastive loss-> {self.contrastive}, conditioned: {self.conditioned}, proj: {self.proj}')

        if self.target == 'pixel':
            recon_feat_dim = patch_size ** 2 * in_chans
        elif self.target == 'hog':
            recon_feat_dim = patch_size ** 2 * 9
        elif self.target == 'mgf':
            recon_feat_dim = patch_size ** 2 * 3
        elif self.target == 'canny':
            recon_feat_dim = patch_size ** 2
        else:
            raise NotImplementedError
        #
        if self.reduction == 'channel' or self.reduction == 'dual':
            recon_feat_dim = patch_size ** 2
        elif self.reduction == 'spatial' or self.reduction == 'none' or self.reduction == 'median' or self.reduction == 'gaussian':
            recon_feat_dim = recon_feat_dim
        else:
            raise NotImplementedError

        self.feature = DualFeat(filter=self.target, reduction=self.reduction, input_size=img_size, patch_size=4)

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        if self.modality == 'RGB' or self.modality == 'Both':
            self.patch_embed_rgb = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed_rgb.num_patches
            Hp, Wp = self.patch_embed_rgb.patches_resolution

        if self.modality == 'SAR' or self.modality == 'Both':
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

        if self.dino:
            dpr_cls = list(x.item() for x in torch.linspace(0, drop_path_rate, 2 * mlp_depth + depth))

        self.blocks = nn.ModuleList()

        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['4'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['4'], norm_layer))
        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['8'], 0, bridge_mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(mlp_depth)]
        )
        self.blocks.append(PatchMerge(mlvl_dims['8'], norm_layer))

        self.blocks.extend([
            BlockWithRPE(
                Hp, mlvl_dims['16'], num_heads, mlp_ratio, qkv_bias, qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                rpe=rpe, norm_layer=norm_layer
            ) for _ in range(depth)]
        )

        ########################### FPN PART ###########################
        if self.num_outs > 1:
            self.align_dim_16tofpn = nn.Linear(embed_dim, fpn_dim) if embed_dim != fpn_dim else None
            self.fpn_modules = nn.ModuleList()
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                ))

            self.align_dim_16to8 = nn.Linear(mlvl_dims['8'], fpn_dim, bias=False)
            self.split_16to8 = PatchSplit(mlvl_dims['16'], fpn_dim, norm_layer)
            self.block_16to8 = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ))

        if self.num_outs > 2:
            self.align_dim_8to4 = nn.Linear(mlvl_dims['4'], fpn_dim, bias=False)
            self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_layer)
            self.block_8to4 = nn.Sequential(
                *[BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer,
                ) for _ in range(fpn_depth)]
            )
            self.fpn_modules.append(
                BlockWithRPE(
                    Hp, fpn_dim, 0, mlp_ratio, qkv_bias, qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.,
                    rpe=rpe, norm_layer=norm_layer
                )
            )

        # --------------------------------------------------------------------------

        ########################### Contrastive Loss ###########################
        if self.contrastive:
            self.contrastive_loss = ContrastiveLoss(in_dim=decoder_embed_dim, out_dim=256, conditioned=self.conditioned,
                                                    proj=self.proj,
                                                    input_size=Hp, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=0., rpe=False, norm_layer=norm_layer)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_size = patch_size

        self.decoder_embed = nn.ModuleList()
        self.decoder_embed.append(
            nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim, bias=True)))
        if self.num_outs >= 2:
            self.decoder_embed.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 4, bias=True)))
        if self.num_outs >= 3:
            self.decoder_embed.append(
                nn.Sequential(norm_layer(fpn_dim), nn.Linear(fpn_dim, decoder_embed_dim // 16, bias=True)))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_blocks = nn.ModuleList([
            BlockWithRPE(
                Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                rpe=False, norm_layer=norm_layer, crossattn=False
            )
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        if self.dino:
            self.lm_head = nn.Linear(decoder_embed_dim, 1024)
            self.lm_cls_head = nn.Linear(embed_dim, 1024)
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

        if self.cr_layers > 0:

            self.mask_token_cross = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            torch.nn.init.normal_(self.mask_token_cross, std=.02)

            self.decoder_blocks_cross = nn.ModuleList([
                BlockWithRPE(
                    Hp, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias, qk_scale,
                    rpe=False, norm_layer=norm_layer, crossattn=False
                )
                for _ in range(decoder_depth)])

            self.decoder_norm_cross = norm_layer(decoder_embed_dim)
            self.decoder_pred_cross = nn.Linear(decoder_embed_dim, recon_feat_dim, bias=True)


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
        if self.modality == 'Both':
            B *= 2

        # Hp, Wp = H // self.patch_embed_rgb.patch_size[0], W // self.patch_embed_rgb.patch_size[1]

        if self.modality == 'RGB':
            x = self.patch_embed_rgb(x_rgb)
            Hp, Wp = H // self.patch_embed_rgb.patch_size[0], W // self.patch_embed_rgb.patch_size[1]
        elif self.modality == 'SAR':
            x = self.patch_embed_sar(x_sar)
            Hp, Wp = H // self.patch_embed_sar.patch_size[0], W // self.patch_embed_sar.patch_size[1]
        elif self.modality == 'Both':
            x = torch.cat([self.patch_embed_rgb(x_rgb), self.patch_embed_sar(x_sar)], dim=0)
            Hp, Wp = H // self.patch_embed_rgb.patch_size[0], W // self.patch_embed_rgb.patch_size[1]
        else:
            raise NotImplementedError

        if ids_keep is not None:
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

        features = []
        for blk in self.blocks[:-self.num_main_blocks]:
            if isinstance(blk, PatchMerge):
                features.append(x)
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.interpolate_pos_encoding(x, Hp, Wp)
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for j, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
            if j == int(0.75 * self.depth):
                if self.dino:
                    aux_out = x
                else:
                    aux_out = None

        if self.num_outs == 1:
            return x

        ##########################  FPN forward  ########################

        x = x[..., None, None, :]
        outs = [x] if self.align_dim_16tofpn is None else [self.align_dim_16tofpn(x)]
        if self.num_outs >= 2:
            x = self.block_16to8(self.split_16to8(x) + self.align_dim_16to8(features[1]))
            outs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(self.split_8to4(x) + self.align_dim_8to4(features[0]))
            outs.append(x)
        if rpe_index is None and self.num_outs > 3:
            outs = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                    B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
                for out in outs]

            if self.num_outs >= 4:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs):
            out = self.fpn_modules[i](out)
            outs[i] = out

        if self.dino:
            for blk in self.cls_pt_layers:
                aux_out = blk(aux_out, rpe_index)

        return outs, aux_out

    def forward_encoder(self, x_rgb, x_sar, mask_ratio):

        ids_keep, ids_restore, mask = self.masking_id(x_rgb.size(0), mask_ratio)
        if self.modality == 'Both':
            ids_keep = torch.cat([ids_keep, ids_keep], dim=0)
            ids_restore = torch.cat([ids_restore, ids_restore], dim=0)
            mask = torch.cat([mask, mask], dim=0)

        x, aux_x = self.forward_features(x_rgb, x_sar, ids_keep=ids_keep)

        return x, aux_x, mask, ids_restore, ids_keep

    def forward_decoder(self, inps, ids_restore):  # latent, ids_restore
        B, N, _, _, _ = inps[0].shape
        half_B = B // 2

        feats = []
        for feat, layer in zip(inps, self.decoder_embed):
            x = layer(feat).reshape(B, N, -1)
            feats.append(x)

        distill_feats = []
        for x in feats:
            distill_feats.append(x)
        x_distill = distill_feats.pop(0)
        for i, feat in enumerate(distill_feats):
            x_distill = x_distill + distill_feats[i]

        if self.modality != 'Both':
            x_rgb_distill = None
            x_sar_distill = None
        else:
            x_rgb_distill = x_distill[:half_B]
            x_sar_distill = x_distill[half_B:]

        # tokens for self-modality reconstruction (RGB)
        feats_self = []
        for x in feats:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            feats_self.append(x)
        x_self = feats_self.pop(0)
        for i, feat in enumerate(feats_self):
            x_self = x_self + feats_self[i]
        x_self = x_self + self.decoder_pos_embed

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            # x_self = blk(x_self)
            x_self = checkpoint.checkpoint(blk, x_self) if self.use_checkpoint else blk(x_self)

        x_self = self.decoder_norm(x_self)
        self_dec = self.decoder_pred(x_self)

        if self.cr_layers > 0:
            # tokens for cross-modality reconstruction (RGB)

            feats_cross = []
            for x in feats:
                mask_tokens = self.mask_token_cross.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                x = torch.cat([x, mask_tokens], dim=1)
                x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                feats_cross.append(x)
            x_cross = feats_cross.pop(0)
            for i, feat in enumerate(feats_cross):
                x_cross = x_cross + feats_cross[i]
            x_cross = x_cross + self.decoder_pos_embed

            for i, blk in enumerate(self.decoder_blocks_cross):
                # x_cross = blk(x_cross)
                x_cross = checkpoint.checkpoint(blk, x_cross) if self.use_checkpoint else blk(x_cross)

            x_cross = self.decoder_norm_cross(x_cross)
            cross_dec = self.decoder_pred_cross(x_cross)

        else:
            cross_dec = torch.zeros_like(self_dec)

        return x_rgb_distill, x_sar_distill, self_dec, cross_dec

    def forward_loss(self, x_rgb, x_sar, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        num_preds = mask.sum()
        if self.modality == 'RGB':
            target = self.patchify(x_rgb)
        elif self.modality == 'SAR':
            target = self.patchify(x_sar)
        elif self.modality == 'Both':
            target = torch.cat([self.patchify(x_rgb), self.patchify(x_sar)], dim=0)
        else:
            raise NotImplementedError

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds

        return loss  # , loss_rgb, loss_sar

    def forward_loss_cross(self, x_rgb, x_sar, pred, mask, sp_norm=False):
        num_preds = mask.sum()
        target = torch.cat([self.patchify(x_rgb), self.patchify(x_sar)], dim=0)

        # we need to keep the relative magnitude across patches when spatial reduction is applied,
        # otherwise we get mean=0 and var=0
        if self.norm_pix_loss:
            if sp_norm:
                mean = target.mean(dim=(1, 2), keepdim=True)
                var = target.var(dim=(1, 2), keepdim=True)
            else:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds

        return loss  # , loss_rgb, loss_sar

    def forward(self, imgs, mask_ratio=0.75, tea_feat=None):
        inputs = imgs['normalized']

        x_rgb = inputs[:, :3, :, :].cuda(non_blocking=True)
        x_sar = inputs[:, 3:, :, :].cuda(non_blocking=True)
        is_paired = torch.all(imgs['paired'] == 1)

        latent, aux_x, mask, ids_restore, ids_keep = self.forward_encoder(x_rgb, x_sar, mask_ratio)

        x_rgb_distill, x_sar_distill, self_pred, cross_pred = self.forward_decoder(latent, ids_restore)

        if self.dino:
            dis_out = self.norm(x_rgb_distill)
            dis_out = self.lm_head(dis_out)
            dis_out_aux = self.norm_cls(aux_x[:x_rgb.shape[0], :, :])
            dis_out_aux = self.lm_cls_head(dis_out_aux)
            tea_feat = torch.gather(tea_feat, dim=1,
                                    index=ids_keep[:x_rgb.shape[0], :, None].expand(-1, -1, tea_feat.shape[2]))
            distill_loss_main = nn.SmoothL1Loss()(input=dis_out, target=tea_feat)
            distill_loss_aux = nn.SmoothL1Loss()(input=dis_out_aux, target=tea_feat)
            distill_loss = (distill_loss_main + distill_loss_aux)  # / 2
        else:
            distill_loss = 0.

        if self.contrastive:
            contrastive_loss = self.contrastive_loss(x_rgb_distill, x_sar_distill)
            if not is_paired:
                contrastive_loss = contrastive_loss * 0.
        else:
            contrastive_loss = 0.

        self_loss = self.forward_loss(x_rgb, x_sar, self_pred, mask)

        if self.cr_layers > 0:
            ft_rgb, ft_sar = self.feature(imgs)
            cross_loss = self.forward_loss_cross(ft_sar, ft_rgb, cross_pred, mask)

            if not is_paired:
                cross_loss = cross_loss * 0.
        else:
            cross_loss = 0.

        return self_loss, cross_loss, distill_loss, contrastive_loss, self_pred, cross_pred, mask

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
                                contrastive=True,
                                conditioned=True,
                                norm_pix_loss=True)

    model.cuda()
    x = torch.randn(4, 6, 224, 224).cuda()
    x = {'normalized': x,
         'paired': torch.tensor([1, 1, 1, 1])}
    loss = model(x)
    print(loss)