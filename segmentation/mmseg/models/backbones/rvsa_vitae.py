"""ViTAE-B with rotated varied-size window attention (RVSA).

This is a compatibility port of the official Remote-Sensing-RVSA backbone to
MMSegmentation 1.x/MMEngine.  Parameter names intentionally follow the
official implementation so that its MillionAID MAE checkpoint can be loaded
without a key conversion script.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint

from mmseg.registry import MODELS


def _to_2tuple(value) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1. - self.drop_prob
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class Mlp(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 drop: float = 0.) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def _calc_rel_pos_spatial(attn: torch.Tensor, q: torch.Tensor,
                          q_shape: Tuple[int, int],
                          k_shape: Tuple[int, int],
                          rel_pos_h: torch.Tensor,
                          rel_pos_w: torch.Tensor) -> torch.Tensor:
    """Add decomposed spatial relative-position embeddings."""
    q_h, q_w = q_shape
    k_h, k_w = k_shape
    device = q.device

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h, device=device)[:, None] * q_h_ratio -
              torch.arange(k_h, device=device)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio

    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w, device=device)[:, None] * q_w_ratio -
              torch.arange(k_w, device=device)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    rel_h = torch.einsum('byhwc,hkc->byhwk',
                         q.reshape(q.shape[0], q.shape[1], q_h, q_w,
                                   q.shape[-1]), rel_pos_h[dist_h.long()])
    rel_w = torch.einsum('byhwc,wkc->byhwk',
                         q.reshape(q.shape[0], q.shape[1], q_h, q_w,
                                   q.shape[-1]), rel_pos_w[dist_w.long()])
    attn = attn.reshape(attn.shape[0], attn.shape[1], q_h, q_w, k_h,
                        k_w)
    attn = attn + rel_h[..., None] + rel_w[..., None, :]
    return attn.reshape(attn.shape[0], attn.shape[1], q_h * q_w,
                        k_h * k_w)


class Attention(nn.Module):
    """Full attention used at every ``interval``-th ViTAE block."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Tuple[int, int],
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.window_size = _to_2tuple(window_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        rel_sp_dim = 2 * self.window_size[0] - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f'Expected {h * w} tokens, but got {n}.')
        if (h, w) != self.window_size:
            raise ValueError(
                'Global RVSA blocks require the runtime patch grid to match '
                f'the configured grid {self.window_size}, but got {(h, w)}.')

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads,
                          c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = _calc_rel_pos_spatial(attn, q, (h, w), (h, w),
                                     self.rel_pos_h, self.rel_pos_w)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class RotatedVariedSizeWindowAttention(nn.Module):
    """Rotated varied-size window attention from the official RVSA code."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Tuple[int, int] = (7, 7),
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 num_deform: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.num_heads = num_heads
        self.num_deform = num_deform
        self.ws = _to_2tuple(window_size)[0]
        self.attn_ws = self.ws
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        rel_sp_dim = 2 * self.ws - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.sampling_offsets = nn.Sequential(
            nn.AvgPool2d(self.ws, self.ws), nn.LeakyReLU(),
            nn.Conv2d(dim, num_heads * num_deform * 2, 1))
        self.sampling_scales = nn.Sequential(
            nn.AvgPool2d(self.ws, self.ws), nn.LeakyReLU(),
            nn.Conv2d(dim, num_heads * num_deform * 2, 1))
        self.sampling_angles = nn.Sequential(
            nn.AvgPool2d(self.ws, self.ws), nn.LeakyReLU(),
            nn.Conv2d(dim, num_heads * num_deform, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        table_size = (self.ws + self.attn_ws - 1)**2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads))
        coords = torch.stack(
            torch.meshgrid(torch.arange(self.attn_ws),
                           torch.arange(self.attn_ws),
                           indexing='ij'))
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.attn_ws - 1
        relative_coords[:, :, 1] += self.attn_ws - 1
        relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
        self.register_buffer(
            'relative_position_index',
            relative_coords.sum(-1),
            persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, tokens: torch.Tensor, h: int,
                w: int) -> torch.Tensor:
        b, n, c = tokens.shape
        if n != h * w:
            raise ValueError(f'Expected {h * w} tokens, but got {n}.')
        x = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        shortcut = x

        padding_h = (self.ws - h % self.ws) % self.ws
        padding_w = (self.ws - w % self.ws) % self.ws
        padding_top = padding_h // 2
        padding_bottom = padding_h - padding_top
        padding_left = padding_w // 2
        padding_right = padding_w - padding_left
        expand_h = h + padding_h
        expand_w = w + padding_w
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws

        coord_dtype = x.dtype if x.dtype in (torch.float32,
                                              torch.float64) else torch.float32
        image_reference_h = torch.linspace(
            -1, 1, expand_h, device=x.device, dtype=coord_dtype)
        image_reference_w = torch.linspace(
            -1, 1, expand_w, device=x.device, dtype=coord_dtype)
        grid_w, grid_h = torch.meshgrid(
            image_reference_w, image_reference_h, indexing='ij')
        image_reference = torch.stack((grid_w, grid_h), 0).permute(
            0, 2, 1).unsqueeze(0)
        window_reference = F.avg_pool2d(image_reference, self.ws)
        image_reference = image_reference.reshape(
            1, 2, window_num_h, self.ws, window_num_w, self.ws)
        window_reference = window_reference.reshape(
            1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = (torch.arange(
            self.attn_ws, device=x.device, dtype=coord_dtype) * 2 * self.ws /
                         self.attn_ws / max(expand_h - 1, 1))
        base_coords_h -= base_coords_h.mean()
        base_coords_w = (torch.arange(
            self.attn_ws, device=x.device, dtype=coord_dtype) * 2 * self.ws /
                         self.attn_ws / max(expand_w - 1, 1))
        base_coords_w -= base_coords_w.mean()
        expanded_h = base_coords_h[None].repeat(window_num_h, 1).reshape(-1)
        expanded_w = base_coords_w[None].repeat(window_num_w, 1).reshape(-1)
        grid_w, grid_h = torch.meshgrid(expanded_w, expanded_h, indexing='ij')
        window_coords = torch.stack((grid_w, grid_h), 0).permute(
            0, 2, 1).reshape(1, 2, window_num_h, self.attn_ws,
                             window_num_w, self.attn_ws)

        x = F.pad(x,
                  (padding_left, padding_right, padding_top, padding_bottom))
        sampling_offsets = self.sampling_offsets(x)
        num_predict_total = b * self.num_heads * self.num_deform
        sampling_offsets = sampling_offsets.reshape(
            num_predict_total, 2, window_num_h, window_num_w)
        sampling_offsets[:, 0] = sampling_offsets[:, 0] / max(h // self.ws, 1)
        sampling_offsets[:, 1] = sampling_offsets[:, 1] / max(w // self.ws, 1)
        sampling_scales = self.sampling_scales(x).reshape(
            num_predict_total, 2, window_num_h, window_num_w)
        sampling_angles = self.sampling_angles(x).reshape(
            num_predict_total, 1, window_num_h, window_num_w)

        window_coords = window_coords * (
            sampling_scales[:, :, :, None, :, None] + 1)
        angle = sampling_angles[:, 0, :, None, :, None]
        rotated_coords = torch.empty_like(window_coords)
        rotated_coords[:, 0] = (-window_coords[:, 1] * torch.sin(angle) +
                                window_coords[:, 0] * torch.cos(angle))
        rotated_coords[:, 1] = (window_coords[:, 1] * torch.cos(angle) +
                                window_coords[:, 0] * torch.sin(angle))
        sample_coords = (window_reference + rotated_coords +
                         sampling_offsets[:, :, :, None, :, None])
        sample_coords = sample_coords.permute(0, 2, 3, 4, 5, 1).reshape(
            num_predict_total, self.attn_ws * window_num_h,
            self.attn_ws * window_num_w, 2)

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(shortcut.permute(0, 2, 3, 1).reshape(b, -1, c),
                       self.qkv.weight, qkv_bias)
        qkv = qkv.permute(0, 2, 1).reshape(b, 3, self.num_heads,
                                           c // self.num_heads, h, w)
        qkv = qkv.transpose(1, 0).reshape(3 * b * self.num_heads,
                                          c // self.num_heads, h, w)
        qkv = F.pad(qkv,
                    (padding_left, padding_right, padding_top, padding_bottom))
        qkv = qkv.reshape(3, b * self.num_heads, c // self.num_heads,
                          expand_h, expand_w)
        q, k, v = qkv.unbind(0)

        per_deform_dim = c // self.num_heads // self.num_deform
        k = F.grid_sample(
            k.reshape(num_predict_total, per_deform_dim, expand_h, expand_w),
            sample_coords,
            padding_mode='zeros',
            align_corners=True).reshape(b * self.num_heads,
                                        c // self.num_heads, expand_h, expand_w)
        v = F.grid_sample(
            v.reshape(num_predict_total, per_deform_dim, expand_h, expand_w),
            sample_coords,
            padding_mode='zeros',
            align_corners=True).reshape(b * self.num_heads,
                                        c // self.num_heads, expand_h, expand_w)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, window_num_h, self.ws,
                      window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2)
        k = k.reshape(b, self.num_heads, head_dim, window_num_h, self.attn_ws,
                      window_num_w,
                      self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2)
        v = v.reshape(b, self.num_heads, head_dim, window_num_h, self.attn_ws,
                      window_num_w,
                      self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2)
        q = q.reshape(-1, self.num_heads, self.ws * self.ws, head_dim)
        k = k.reshape(-1, self.num_heads, self.attn_ws * self.attn_ws,
                      head_dim)
        v = v.reshape(-1, self.num_heads, self.attn_ws * self.attn_ws,
                      head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = _calc_rel_pos_spatial(attn, q, (self.ws, self.ws),
                                     (self.attn_ws, self.attn_ws),
                                     self.rel_pos_h, self.rel_pos_w)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)]
        relative_bias = relative_bias.reshape(
            self.ws * self.ws, self.attn_ws * self.attn_ws,
            self.num_heads).permute(2, 0, 1)
        attn = self.attn_drop((attn + relative_bias.unsqueeze(0)).softmax(-1))
        out = attn @ v
        out = out.reshape(b, window_num_h, window_num_w, self.num_heads,
                          self.ws, self.ws, head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).reshape(
            b, c, expand_h, expand_w)
        out = out[:, :, padding_top:padding_top + h,
                  padding_left:padding_left + w]
        out = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.proj_drop(self.proj(out))


def _make_pcm_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(channels)
    if norm_type == 'BN':
        return nn.BatchNorm2d(channels)
    raise ValueError("pcm_norm must be either 'BN' or 'SyncBN'.")


class NormalCell(nn.Module):
    """ViTAE normal cell: attention + parallel convolution + MLP."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float,
                 window_size: Tuple[int, int],
                 window: bool,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 pcm_norm: str = 'BN') -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        attention_cls = RotatedVariedSizeWindowAttention if window else Attention
        self.attn = attention_cls(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_dim, drop=drop)

        # Keep the official uppercase attribute and indices for checkpoint keys.
        groups = dim // 4
        self.PCM = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=groups),
            _make_pcm_norm(pcm_norm, hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 3, 1, 1, groups=groups))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x_2d = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        conv_x = self.PCM(x_2d).permute(0, 2, 3, 1).reshape(b, n, c)
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(conv_x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):

    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 embed_dims: int) -> None:
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.patch_shape = (self.img_size[0] // self.patch_size[0],
                            self.img_size[1] // self.patch_size[1])
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        h, w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (h, w)


class Norm2d(nn.Module):

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dims, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()


@MODELS.register_module()
class ViTAE_NC_Win_RVSA_V3_WSZ7(BaseModule):
    """ViTAE-B + RVSA backbone for dense prediction.

    The Base configuration is 12 blocks, 768 channels, 12 heads, patch size
    16, RVSA window size 7, and a full-attention block every third block.
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = .1,
                 use_checkpoint: bool = False,
                 use_abs_pos_emb: bool = True,
                 out_indices: Sequence[int] = (3, 5, 7, 11),
                 interval: int = 3,
                 pcm_norm: str = 'BN',
                 interpolate_pos_embed: bool = True,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if patch_size != 16:
            raise ValueError('This RVSA port currently supports patch_size=16.')
        if len(out_indices) != 4:
            raise ValueError('Exactly four out_indices are required by the FPN.')
        if embed_dims % num_heads != 0 or embed_dims % 4 != 0:
            raise ValueError('embed_dims must be divisible by num_heads and 4.')

        self.embed_dims = embed_dims
        self.depth = depth
        self.out_indices = tuple(out_indices)
        self.use_checkpoint = use_checkpoint
        self.interpolate_pos_embed = interpolate_pos_embed
        self.pretrained = pretrained
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels,
                                      embed_dims)
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dims))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            is_window = (i + 1) % interval != 0
            window_size = (7, 7) if is_window else self.patch_embed.patch_shape
            self.blocks.append(
                NormalCell(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size,
                    window=is_window,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pcm_norm=pcm_norm))
        # Retained for exact compatibility with the official pretraining keys.
        self.norm = nn.LayerNorm(embed_dims, eps=1e-6)

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2),
            Norm2d(embed_dims), nn.GELU(),
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2))
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dims, embed_dims,
                                                     2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _checkpoint_path(self) -> Optional[str]:
        if self.pretrained is not None:
            return self.pretrained
        if self.init_cfg is None:
            return None
        init_cfg = self.init_cfg[0] if isinstance(self.init_cfg,
                                                  list) else self.init_cfg
        if init_cfg.get('type') != 'Pretrained':
            return None
        return init_cfg.get('checkpoint')

    def _resize_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        source_tokens = pos_embed.shape[1]
        source_size = int(math.sqrt(source_tokens))
        if source_size * source_size != source_tokens:
            source_tokens -= 1
            source_size = int(math.sqrt(source_tokens))
            if source_size * source_size != source_tokens:
                raise ValueError(
                    f'Cannot infer a square grid from pos_embed {pos_embed.shape}.')
            pos_embed = pos_embed[:, 1:]
        target_h, target_w = self.patch_embed.patch_shape
        if (source_size, source_size) == (target_h, target_w):
            return pos_embed
        pos_embed = pos_embed.reshape(1, source_size, source_size,
                                      pos_embed.shape[-1]).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False)
        return pos_embed.permute(0, 2, 3, 1).reshape(
            1, target_h * target_w, self.embed_dims)

    def _adapt_checkpoint(self, checkpoint_dict: Dict) -> Dict[str,
                                                               torch.Tensor]:
        state_dict = checkpoint_dict
        for key in ('state_dict', 'model'):
            if isinstance(state_dict, dict) and key in state_dict and isinstance(
                    state_dict[key], dict):
                state_dict = state_dict[key]
                break
        model_state = self.state_dict()
        adapted = {}
        pending_qkv_bias = {}
        for raw_name, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            name = raw_name
            for prefix in ('module.', 'backbone.'):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            if name == 'pos_embed' and self.pos_embed is not None:
                if self.interpolate_pos_embed:
                    value = self._resize_pos_embed(value)
            if name.endswith('.attn.qkv.bias'):
                pending_qkv_bias[name[:-len('qkv.bias')]] = value
                continue
            if name in model_state and model_state[name].shape == value.shape:
                adapted[name] = value

        # Also accept the qkv-bias layout used by the plain ViT-B checkpoint.
        for prefix, bias in pending_qkv_bias.items():
            q_bias, _, v_bias = bias.chunk(3)
            q_name = prefix + 'q_bias'
            v_name = prefix + 'v_bias'
            if q_name in model_state and model_state[q_name].shape == q_bias.shape:
                adapted[q_name] = q_bias
                adapted[v_name] = v_bias
        return adapted

    def init_weights(self) -> None:
        self.apply(self._init_module)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None:
            print_log('RVSA ViTAE-B initialized without a pretrained checkpoint.',
                      logger='current')
            return

        checkpoint_dict = _load_checkpoint(
            checkpoint_path, map_location='cpu', logger='current')
        adapted = self._adapt_checkpoint(checkpoint_dict)
        incompatible = self.load_state_dict(adapted, strict=False)
        print_log(
            'Loaded RVSA ViTAE-B pretraining weights from '
            f'{checkpoint_path}: {len(adapted)} tensors loaded, '
            f'{len(incompatible.missing_keys)} newly initialized tensors.',
            logger='current')

    def get_num_layers(self) -> int:
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def _position_embedding(self, h: int, w: int) -> torch.Tensor:
        if (h, w) != self.patch_embed.patch_shape:
            raise ValueError(
                'RVSA global-attention relative-position parameters are tied '
                f'to patch grid {self.patch_embed.patch_shape}; got {(h, w)}. '
                'Pad/resize inputs to the configured img_size.')
        return self.pos_embed

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        x, (h, w) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self._position_embedding(h, w)
        x = self.pos_drop(x)

        features = []
        for index, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    lambda inputs, current_block=block: current_block(
                        inputs, h, w),
                    x,
                    use_reentrant=False)
            else:
                x = block(x, h, w)
            if index in self.out_indices:
                features.append(
                    x.transpose(1, 2).reshape(b, self.embed_dims, h, w))

        operations = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        return tuple(operation(feature)
                     for operation, feature in zip(operations, features))
