"""SARATR-X HiViT backbone adapter for MMSegmentation."""

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
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS


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


def _window_unpartition(windows, window_size, padded_shape, output_shape):
    """Reverse :func:`_window_partition` and remove its padding."""
    padded_height, padded_width = padded_shape
    height, width = output_shape
    batch_size = windows.shape[0] // (
        padded_height * padded_width // window_size**2)
    outputs = windows.view(
        batch_size, padded_height // window_size,
        padded_width // window_size, window_size, window_size, -1)
    outputs = outputs.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        batch_size, padded_height, padded_width, -1)
    if padded_height > height or padded_width > width:
        outputs = outputs[:, :height, :width].contiguous()
    return outputs


class SARATRXLayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW feature map."""

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = super().forward(inputs)
        return inputs.permute(0, 3, 1, 2).contiguous()


class SARATRXMLP(nn.Module):
    """MLP retaining timm's checkpoint parameter names."""

    def __init__(self, embed_dims, hidden_dims, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dims, embed_dims)

    def forward(self, inputs):
        inputs = self.drop(self.act(self.fc1(inputs)))
        return self.drop(self.fc2(inputs))


class SARATRXAttention(nn.Module):
    """The attention implementation used by the official HiViT."""

    def __init__(self,
                 input_size,
                 embed_dims,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 drop_rate=0.,
                 rpe=True):
        super().__init__()
        self.num_heads = num_heads
        head_dims = embed_dims // num_heads
        self.scale = qk_scale or head_dims**-0.5
        if rpe:
            table_size = (2 * input_size - 1)**2
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(table_size, num_heads))
        else:
            self.register_parameter('relative_position_bias_table', None)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, inputs, relative_position_index=None, mask=None):
        batch_size, num_tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size, num_tokens, 3, self.num_heads,
            channels // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        if relative_position_index is not None:
            side = int(math.sqrt(relative_position_index.numel()))
            relative_bias = self.relative_position_bias_table[
                relative_position_index].view(
                    -1, side, side, self.num_heads)
            relative_bias = relative_bias.permute(0, 3, 1, 2).contiguous()
            if num_tokens != side:
                raise ValueError('Relative-position index has wrong size.')
            attention = attention + relative_bias
        if mask is not None:
            attention = attention.masked_fill(
                ~mask.bool()[:, None, None, :], float('-inf'))
        attention = attention.float().clamp(
            min=torch.finfo(torch.float32).min,
            max=torch.finfo(torch.float32).max)
        attention = self.attn_drop(attention.softmax(dim=-1))
        outputs = (attention @ value).transpose(1, 2).reshape(
            batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(outputs))


class SARATRXBlock(nn.Module):
    """An attention block, or an MLP-only HiViT stem block."""

    def __init__(self,
                 input_size,
                 embed_dims,
                 num_heads=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 rpe=True):
        super().__init__()
        with_attention = num_heads > 0
        self.norm1 = nn.LayerNorm(
            embed_dims, eps=1e-6) if with_attention else None
        self.attn = SARATRXAttention(
            input_size=input_size,
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            rpe=rpe) if with_attention else None
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.mlp = SARATRXMLP(
            embed_dims, int(embed_dims * mlp_ratio), drop_rate)

    def forward(self, inputs, relative_position_index=None, mask=None):
        if self.attn is not None:
            inputs = inputs + self.drop_path(
                self.attn(
                    self.norm1(inputs), relative_position_index, mask))
        return inputs + self.drop_path(self.mlp(self.norm2(inputs)))


class SARATRXPatchEmbed(nn.Module):
    """HiViT's inner-patch embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 inner_patches=4,
                 in_channels=3,
                 embed_dims=128,
                 patch_norm=True):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.inner_patches = inner_patches
        self.patches_resolution = tuple(
            size // patch for size, patch in zip(
                self.img_size, self.patch_size))
        self.num_patches = math.prod(self.patches_resolution)
        conv_size = tuple(
            patch // inner_patches for patch in self.patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=conv_size,
            stride=conv_size)
        self.norm = nn.LayerNorm(
            embed_dims, eps=1e-6) if patch_norm else None

    def forward(self, inputs):
        batch_size, _, height, width = inputs.shape
        patch_height = height // self.patch_size[0]
        patch_width = width // self.patch_size[1]
        outputs = self.proj(inputs).view(
            batch_size, -1, patch_height, self.inner_patches,
            patch_width, self.inner_patches)
        outputs = outputs.permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size, patch_height * patch_width,
            self.inner_patches, self.inner_patches, -1)
        return self.norm(outputs) if self.norm is not None else outputs


class SARATRXPatchMerge(nn.Module):
    """Merge a 2x2 region inside each hierarchical patch."""

    def __init__(self, embed_dims):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims * 4, eps=1e-6)
        self.reduction = nn.Linear(
            embed_dims * 4, embed_dims * 2, bias=False)

    def forward(self, inputs):
        top_left = inputs[..., 0::2, 0::2, :]
        bottom_left = inputs[..., 1::2, 0::2, :]
        top_right = inputs[..., 0::2, 1::2, :]
        bottom_right = inputs[..., 1::2, 1::2, :]
        outputs = torch.cat(
            (top_left, bottom_left, top_right, bottom_right), dim=-1)
        return self.reduction(self.norm(outputs))


@MODELS.register_module()
class SARATRXHiViT(BaseModule):
    """SARATR-X HiViT-B encoder adapted for semantic segmentation.

    This follows the official detection transfer model rather than treating
    the checkpoint as a conventional ViT. The two MLP-only stem stages retain
    4x4 and 2x2 inner-patch structure before the 20-block attention stage.
    The released checkpoint's predictor and SAR-gradient targets are used
    only during masked pre-training and are intentionally ignored here.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 inner_patches=4,
                 in_channels=3,
                 embed_dims=512,
                 depths=(2, 2, 20),
                 num_heads=8,
                 stem_mlp_ratio=3.,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 ape=True,
                 rpe=False,
                 patch_norm=True,
                 out_indices=('H', 'M', 19, 19),
                 global_indices=(4, 9, 14, 19),
                 window_size=14,
                 with_cp=False,
                 expand_single_channel=True,
                 strict_encoder_load=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if patch_size != 16 or inner_patches != 4:
            raise ValueError(
                'Released SARATR-X HiViT requires patch_size=16 and '
                'inner_patches=4.')
        if len(depths) != 3 or len(out_indices) != 4:
            raise ValueError('SARATR-X requires three stages and four outs.')
        if out_indices.count('H') != 1 or out_indices.count('M') != 1:
            raise ValueError('Official dense transfer requires H and M outs.')

        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.depths = tuple(depths)
        self.num_main_blocks = depths[-1]
        self.ape = ape
        self.rpe = rpe
        self.out_indices = tuple(out_indices)
        self.global_indices = tuple(global_indices)
        self.window_size = window_size
        self.with_cp = with_cp
        self.expand_single_channel = expand_single_channel
        self.strict_encoder_load = strict_encoder_load

        stem_dims = embed_dims // 2**(len(depths) - 1)
        self.patch_embed = SARATRXPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            in_channels=in_channels,
            embed_dims=stem_dims,
            patch_norm=patch_norm)
        patch_height, patch_width = self.patch_embed.patches_resolution
        if patch_height != patch_width:
            raise ValueError('The released absolute embedding is square.')

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dims))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.register_parameter('absolute_pos_embed', None)

        if rpe:
            relative_position_index = self._build_relative_position_index(
                patch_height, patch_width)
            self.register_buffer(
                'relative_position_index', relative_position_index)
        else:
            self.register_buffer(
                'relative_position_index', None, persistent=False)
        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_rates = iter(torch.linspace(
            0, drop_path_rate,
            sum(depths) + sum(depths[:-1])).tolist())
        blocks = []
        current_dims = stem_dims
        for stage_depth in depths:
            is_main_stage = current_dims == embed_dims
            heads = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            actual_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(actual_depth):
                blocks.append(
                    SARATRXBlock(
                        input_size=patch_height,
                        embed_dims=current_dims,
                        num_heads=heads,
                        mlp_ratio=ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=next(drop_path_rates),
                        rpe=rpe))
            if not is_main_stage:
                blocks.append(SARATRXPatchMerge(current_dims))
                current_dims *= 2
        self.blocks = nn.ModuleList(blocks)

        # Indices deliberately match the official dense-transfer wrapper.
        self.merge_indices = (depths[0] - 1, depths[0] + depths[1])
        self.fpn1 = SARATRXLayerNorm2d(embed_dims // 4, eps=1e-6)
        self.fpn2 = SARATRXLayerNorm2d(embed_dims // 2, eps=1e-6)
        self.fpn3 = SARATRXLayerNorm2d(embed_dims, eps=1e-6)
        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            SARATRXLayerNorm2d(embed_dims, eps=1e-6))

        self.apply(self._init_weights)

    @staticmethod
    def _build_relative_position_index(height, width):
        coords_height = torch.arange(height)
        coords_width = torch.arange(width)
        coords = torch.stack(torch.meshgrid(
            coords_height, coords_width, indexing='ij'))
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += height - 1
        relative_coords[:, :, 1] += width - 1
        relative_coords[:, :, 0] *= 2 * width - 1
        return relative_coords.sum(-1)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.)

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
        if self.init_cfg is None:
            print_log('SARATR-X HiViT-B initialized without pretraining.')
            return
        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        if (init_cfg.get('type') != 'Pretrained' or
                'checkpoint' not in init_cfg):
            raise ValueError(
                'SARATRXHiViT requires a Pretrained init_cfg with checkpoint.')

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
        encoder_prefixes = (
            'absolute_pos_embed', 'patch_embed.', 'blocks.')
        required = {
            key for key in target
            if key == encoder_prefixes[0] or key.startswith(
                encoder_prefixes[1:])
        }
        loadable = OrderedDict()
        incompatible_source = []
        for key, value in source.items():
            if not (key == encoder_prefixes[0] or key.startswith(
                    encoder_prefixes[1:])):
                continue
            if key not in target or value.shape != target[key].shape:
                incompatible_source.append(key)
            else:
                loadable[key] = value
        missing = sorted(required.difference(loadable))
        if self.strict_encoder_load and (missing or incompatible_source):
            raise RuntimeError(
                'SARATR-X encoder checkpoint is not an exact match: '
                f'{len(missing)} missing and {len(incompatible_source)} '
                f'incompatible keys. First missing: {missing[:5]}; first '
                f'incompatible: {incompatible_source[:5]}')

        msg = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SARATR-X encoder tensors from '
            f'{checkpoint_path}; ignored the pre-training norm, predictor '
            'and SAR-gradient target tensors. Pyramid norms remain newly '
            'initialized.')
        print_log(msg)

        del checkpoint, source, loadable
        gc.collect()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def _interpolate_pos_encoding(self, tokens, height, width):
        num_tokens = tokens.shape[1]
        num_positions = self.absolute_pos_embed.shape[1]
        if num_tokens == num_positions and height == width:
            return self.absolute_pos_embed
        base_side = int(math.sqrt(num_positions))
        output_width = width // self.patch_size + 0.1
        output_height = height // self.patch_size + 0.1
        position = self.absolute_pos_embed.reshape(
            1, base_side, base_side, self.embed_dims).permute(0, 3, 1, 2)
        position = F.interpolate(
            position,
            scale_factor=(output_height / base_side,
                          output_width / base_side),
            mode='bicubic')
        if (int(output_height) != position.shape[-2] or
                int(output_width) != position.shape[-1]):
            raise RuntimeError('Absolute-position interpolation failed.')
        return position.permute(0, 2, 3, 1).reshape(
            1, -1, self.embed_dims)

    def _adapt_input_channels(self, inputs):
        if inputs.shape[1] == self.in_channels:
            return inputs
        if (self.expand_single_channel and self.in_channels == 3 and
                inputs.shape[1] == 1):
            return inputs.repeat(1, 3, 1, 1)
        raise ValueError(
            f'Expected {self.in_channels} input channels, got '
            f'{inputs.shape[1]}.')

    def _run_block(self, block, inputs, relative_position_index=None):
        if relative_position_index is None:
            if self.with_cp and self.training:
                return checkpointing.checkpoint(
                    block, inputs, use_reentrant=False)
            return block(inputs)
        if self.with_cp and self.training:
            return checkpointing.checkpoint(
                block,
                inputs,
                relative_position_index,
                use_reentrant=False)
        return block(inputs, relative_position_index)

    def forward(self, inputs):
        inputs = self._adapt_input_channels(inputs)
        batch_size, _, height, width = inputs.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError('Input height and width must be divisible by 16.')
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        features = []
        tokens = self.patch_embed(inputs)
        stem_blocks = self.blocks[:-self.num_main_blocks]
        for index, block in enumerate(stem_blocks):
            tokens = self._run_block(block, tokens)
            if index == self.merge_indices[0]:
                high = tokens.reshape(
                    batch_size, patch_height, patch_width, 4, 4,
                    -1).permute(0, 5, 1, 3, 2, 4).reshape(
                        batch_size, -1, patch_height * 4,
                        patch_width * 4).contiguous()
                features.append(high)
            if index == self.merge_indices[1]:
                medium = tokens.reshape(
                    batch_size, patch_height, patch_width, 2, 2,
                    -1).permute(0, 5, 1, 3, 2, 4).reshape(
                        batch_size, -1, patch_height * 2,
                        patch_width * 2).contiguous()
                features.append(medium)

        tokens = tokens[..., 0, 0, :]
        if self.ape:
            tokens = tokens + self._interpolate_pos_encoding(
                tokens, height, width)
        tokens = self.pos_drop(tokens)
        relative_position_index = (
            self.relative_position_index.flatten() if self.rpe else None)

        main_blocks = self.blocks[-self.num_main_blocks:]
        for index, block in enumerate(main_blocks):
            window_attention = (
                self.window_size is not None and
                index not in self.global_indices)
            if window_attention:
                windows, padded_shape = _window_partition(
                    tokens.reshape(
                        batch_size, patch_height, patch_width, -1),
                    self.window_size)
                tokens = windows.flatten(1, 2)
            tokens = self._run_block(
                block, tokens, relative_position_index)
            if window_attention:
                tokens = _window_unpartition(
                    tokens.reshape(
                        tokens.shape[0], self.window_size,
                        self.window_size, -1),
                    self.window_size,
                    padded_shape,
                    (patch_height, patch_width)).flatten(1, 2)
            if index in self.out_indices:
                low = tokens.transpose(1, 2).reshape(
                    batch_size, -1, patch_height,
                    patch_width).contiguous()
                for _ in range(self.out_indices.count(index)):
                    features.append(low)

        if len(features) != 4:
            raise RuntimeError(f'Expected four features, got {len(features)}.')
        operations = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        return tuple(operation(feature)
                     for operation, feature in zip(operations, features))
