"""SUMMIT ViT-B adapter using the ViTDet dense-prediction architecture."""

import gc
import math
import os.path as osp
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpointing
from mmcv.cnn.bricks import DropPath
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS


def _get_abs_pos(position, has_cls_token, output_size):
    """Drop the pre-training class position and resize the spatial grid."""
    if has_cls_token:
        position = position[:, 1:]
    source_tokens = position.shape[1]
    source_size = int(math.sqrt(source_tokens))
    if source_size * source_size != source_tokens:
        raise ValueError(
            f'Cannot infer a square position grid from {source_tokens} tokens.')

    output_height, output_width = output_size
    if (source_size, source_size) != (output_height, output_width):
        position = position.reshape(
            1, source_size, source_size,
            position.shape[-1]).permute(0, 3, 1, 2)
        position = F.interpolate(
            position,
            size=(output_height, output_width),
            mode='bicubic',
            align_corners=False)
        return position.permute(0, 2, 3, 1)
    return position.reshape(1, output_height, output_width, -1)


def _get_rel_pos(query_size, key_size, relative_position):
    """Resize and index one axis of decomposed relative position."""
    max_distance = 2 * max(query_size, key_size) - 1
    if relative_position.shape[0] != max_distance:
        relative_position = F.interpolate(
            relative_position.transpose(0, 1).unsqueeze(0),
            size=max_distance,
            mode='linear').squeeze(0).transpose(0, 1)

    query_coords = torch.arange(
        query_size, device=relative_position.device)[:, None]
    key_coords = torch.arange(
        key_size, device=relative_position.device)[None, :]
    query_coords = query_coords * max(key_size / query_size, 1.)
    key_coords = key_coords * max(query_size / key_size, 1.)
    relative_coords = (
        query_coords - key_coords +
        (key_size - 1) * max(query_size / key_size, 1.))
    return relative_position[relative_coords.long()]


def _add_decomposed_rel_pos(attention, query, relative_height,
                            relative_width, query_size, key_size):
    """Add ViTDet's decomposed height/width relative position bias."""
    query_height, query_width = query_size
    key_height, key_width = key_size
    height_position = _get_rel_pos(
        query_height, key_height, relative_height)
    width_position = _get_rel_pos(query_width, key_width, relative_width)

    batch_heads, _, head_dims = query.shape
    spatial_query = query.reshape(
        batch_heads, query_height, query_width, head_dims)
    height_bias = torch.einsum(
        'bhwc,hkc->bhwk', spatial_query, height_position)
    width_bias = torch.einsum(
        'bhwc,wkc->bhwk', spatial_query, width_position)
    attention = attention.view(
        batch_heads, query_height, query_width, key_height, key_width)
    attention = attention + height_bias[..., None] + width_bias[..., None, :]
    return attention.view(
        batch_heads, query_height * query_width, key_height * key_width)


def _window_partition(inputs, window_size):
    """Partition an NHWC feature map into padded non-overlapping windows."""
    batch_size, height, width, channels = inputs.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        inputs = F.pad(inputs, (0, 0, 0, pad_width, 0, pad_height))
    padded_height = height + pad_height
    padded_width = width + pad_width
    inputs = inputs.view(
        batch_size, padded_height // window_size, window_size,
        padded_width // window_size, window_size, channels)
    windows = inputs.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, channels)
    return windows, (padded_height, padded_width)


def _window_unpartition(windows, window_size, padded_size, output_size):
    """Reverse :func:`_window_partition` and remove padding."""
    padded_height, padded_width = padded_size
    output_height, output_width = output_size
    batch_size = windows.shape[0] // (
        padded_height * padded_width // window_size**2)
    outputs = windows.view(
        batch_size, padded_height // window_size,
        padded_width // window_size, window_size, window_size, -1)
    outputs = outputs.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        batch_size, padded_height, padded_width, -1)
    if padded_height > output_height or padded_width > output_width:
        outputs = outputs[:, :output_height, :output_width].contiguous()
    return outputs


class SUMMITPatchEmbed(nn.Module):
    """Patch embedding retaining the official ``patch_embed.proj`` names."""

    def __init__(self, patch_size, in_channels, embed_dims):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, inputs):
        return self.proj(inputs).permute(0, 2, 3, 1)


class SUMMITAttention(nn.Module):
    """ViTDet attention with optional decomposed relative position."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 input_size=None):
        super().__init__()
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads.')
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.use_rel_pos = use_rel_pos

        if use_rel_pos:
            if input_size is None:
                raise ValueError('input_size is required with relative position.')
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, self.head_dims))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, self.head_dims))
            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=.02)

    def forward(self, inputs):
        batch_size, height, width, channels = inputs.shape
        num_tokens = height * width
        qkv = self.qkv(inputs).reshape(
            batch_size, num_tokens, 3, self.num_heads,
            self.head_dims).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.reshape(
            3, batch_size * self.num_heads, num_tokens,
            self.head_dims).unbind(0)

        attention = (query * self.scale) @ key.transpose(-2, -1)
        if self.use_rel_pos:
            attention = _add_decomposed_rel_pos(
                attention, query, self.rel_pos_h, self.rel_pos_w,
                (height, width), (height, width))
        attention = attention.softmax(dim=-1)
        outputs = (attention @ value).view(
            batch_size, self.num_heads, height, width,
            self.head_dims).permute(0, 2, 3, 1, 4).reshape(
                batch_size, height, width, channels)
        return self.proj(outputs)


class SUMMITMLP(nn.Module):
    """Timm-compatible two-layer transformer MLP."""

    def __init__(self, embed_dims, hidden_dims, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dims, embed_dims)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, inputs):
        inputs = self.drop1(self.act(self.fc1(inputs)))
        return self.drop2(self.fc2(inputs))


class SUMMITBlock(nn.Module):
    """ViTDet block whose transferable names match the SUMMIT encoder."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 input_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        attention_size = ((window_size, window_size)
                          if window_size else input_size)
        self.attn = SUMMITAttention(
            embed_dims,
            num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=attention_size)
        self.drop_path = (DropPath(drop_path_rate)
                          if drop_path_rate > 0. else nn.Identity())
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.mlp = SUMMITMLP(
            embed_dims, int(embed_dims * mlp_ratio), drop_rate=drop_rate)
        self.window_size = window_size

    def forward(self, inputs):
        shortcut = inputs
        outputs = self.norm1(inputs)
        if self.window_size:
            height, width = outputs.shape[1:3]
            outputs, padded_size = _window_partition(
                outputs, self.window_size)
        outputs = self.attn(outputs)
        if self.window_size:
            outputs = _window_unpartition(
                outputs, self.window_size, padded_size, (height, width))
        outputs = shortcut + self.drop_path(outputs)
        return outputs + self.drop_path(self.mlp(self.norm2(outputs)))


@MODELS.register_module()
class SUMMITViTDet(BaseModule):
    """SUMMIT ViT-B encoder in the MMDetection 3.2 ViTDet form.

    SUMMIT's denoising, edge/scattering reconstruction and ATCM are
    pre-training objectives. The released downstream encoder is an ordinary
    MAE ViT-B. Following the paper's comparison protocol, this class converts
    it to ViTDet: class-token-free spatial features, window attention in eight
    blocks, global attention in blocks 2/5/8/11, and newly initialized
    decomposed relative positions.
    """

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos=True,
                 use_rel_pos=True,
                 rel_pos_zero_init=True,
                 window_size=14,
                 window_block_indices=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=448,
                 pretrain_use_cls_token=True,
                 with_cp=False,
                 strict_encoder_load=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if patch_size != 16:
            raise ValueError('The released SUMMIT model requires patch_size=16.')
        if img_size % patch_size or pretrain_img_size % patch_size:
            raise ValueError('Image sizes must be divisible by patch_size.')

        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.with_cp = with_cp
        self.strict_encoder_load = strict_encoder_load
        self.patch_embed = SUMMITPatchEmbed(
            patch_size, in_channels, embed_dims)

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size)**2
            num_positions = num_patches + int(pretrain_use_cls_token)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dims))
        else:
            self.register_parameter('pos_embed', None)

        input_size = (img_size // patch_size, img_size // patch_size)
        window_block_indices = set(window_block_indices)
        drop_path_rates = torch.linspace(
            0, drop_path_rate, num_layers).tolist()
        self.blocks = nn.ModuleList([
            SUMMITBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rates[index],
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=(window_size
                             if index in window_block_indices else 0),
                input_size=input_size) for index in range(num_layers)
        ])

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _load_checkpoint(checkpoint):
        if isinstance(checkpoint, str) and osp.isfile(checkpoint):
            try:
                return torch.load(checkpoint, map_location='cpu', mmap=True)
            except (RuntimeError, TypeError):
                return torch.load(checkpoint, map_location='cpu')
        return CheckpointLoader.load_checkpoint(
            checkpoint, logger=None, map_location='cpu')

    def init_weights(self):
        self.apply(self._initialize_module)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.init_cfg is None:
            print_log('SUMMIT ViTDet-B initialized without pretraining.')
            return

        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        if (init_cfg.get('type') != 'Pretrained' or
                'checkpoint' not in init_cfg):
            raise ValueError(
                'SUMMITViTDet requires a Pretrained init_cfg with checkpoint.')

        checkpoint_path = init_cfg['checkpoint']
        checkpoint = self._load_checkpoint(checkpoint_path)
        if 'state_dict' in checkpoint:
            source = checkpoint['state_dict']
        elif 'model' in checkpoint:
            source = checkpoint['model']
        else:
            source = checkpoint
        source = OrderedDict(
            (key[7:] if key.startswith('module.') else key, value)
            for key, value in source.items())

        target = self.state_dict()
        required = {
            key for key in target
            if ((key == 'pos_embed' or key.startswith('patch_embed.') or
                 key.startswith('blocks.')) and
                '.rel_pos_h' not in key and '.rel_pos_w' not in key)
        }
        loadable = OrderedDict()
        incompatible_source = []
        for key, value in source.items():
            if not (key == 'pos_embed' or key.startswith('patch_embed.') or
                    key.startswith('blocks.')):
                continue
            if key not in target or value.shape != target[key].shape:
                incompatible_source.append(key)
            else:
                loadable[key] = value

        missing = sorted(required.difference(loadable))
        if self.strict_encoder_load and (missing or incompatible_source):
            raise RuntimeError(
                'SUMMIT encoder checkpoint is not an exact match: '
                f'{len(missing)} missing and {len(incompatible_source)} '
                f'incompatible keys. First missing: {missing[:5]}; first '
                f'incompatible: {incompatible_source[:5]}')
        msg = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SUMMIT encoder tensors from '
            f'{checkpoint_path}; ignored cls_token, final norm and the MAE '
            'decoder. ViTDet relative positions remain newly initialized.')
        print_log(msg)

        del checkpoint, source, loadable
        gc.collect()

    def get_num_layers(self):
        return self.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, inputs):
        if inputs.shape[-2] % self.patch_size or inputs.shape[-1] % self.patch_size:
            raise ValueError('Input height and width must be divisible by 16.')
        features = self.patch_embed(inputs)
        if self.pos_embed is not None:
            features = features + _get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token,
                features.shape[1:3])
        for block in self.blocks:
            if self.with_cp and self.training:
                features = checkpointing.checkpoint(
                    block, features, use_reentrant=False)
            else:
                features = block(features)
        return features.permute(0, 3, 1, 2).contiguous()


class SUMMITLayerNorm2d(nn.Module):
    """ViTDet's channel-wise LayerNorm for NCHW feature maps."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(1, keepdim=True)
        outputs = (inputs - mean) / torch.sqrt(variance + self.eps)
        return (self.weight[:, None, None] * outputs +
                self.bias[:, None, None])


def _conv_norm(in_channels, out_channels, kernel_size, padding=0):
    """ConvModule equivalent used by MMDetection 3.2 SimpleFPN."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False), SUMMITLayerNorm2d(out_channels))


@MODELS.register_module()
class SUMMITSimpleFPN(BaseModule):
    """MMDetection 3.2 SimpleFPN adapted to the MMSeg neck interface."""

    def __init__(self,
                 backbone_channel=768,
                 in_channels=(192, 384, 768, 768),
                 out_channels=256,
                 num_outs=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if len(in_channels) != 4 or num_outs < 4:
            raise ValueError('SUMMITSimpleFPN requires four base levels.')
        self.num_outs = num_outs
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                backbone_channel, backbone_channel // 2, 2, 2),
            SUMMITLayerNorm2d(backbone_channel // 2), nn.GELU(),
            nn.ConvTranspose2d(
                backbone_channel // 2, backbone_channel // 4, 2, 2))
        self.fpn2 = nn.ConvTranspose2d(
            backbone_channel, backbone_channel // 2, 2, 2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lateral_convs = nn.ModuleList([
            _conv_norm(channels, out_channels, 1)
            for channels in in_channels
        ])
        self.output_convs = nn.ModuleList([
            _conv_norm(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])

    def forward(self, inputs):
        features = [
            self.fpn1(inputs),
            self.fpn2(inputs),
            self.fpn3(inputs),
            self.fpn4(inputs),
        ]
        outputs = [
            output_conv(lateral_conv(feature))
            for feature, lateral_conv, output_conv in zip(
                features, self.lateral_convs, self.output_convs)
        ]
        while len(outputs) < self.num_outs:
            outputs.append(F.max_pool2d(outputs[-1], 1, stride=2))
        return tuple(outputs)
