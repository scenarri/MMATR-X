"""SAR-JEPA ViT backbone adapter for MMSegmentation."""

import gc
import math
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpointing
from mmcv.cnn.bricks import DropPath
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS


@torch.no_grad()
def _piecewise_index(relative_position, alpha, beta, gamma):
    """Official iRPE piecewise distance quantization."""
    absolute = relative_position.abs()
    outside = absolute > alpha
    outside_position = relative_position[outside]
    outside_absolute = absolute[outside]
    magnitude = (
        alpha + torch.log(outside_absolute / alpha) /
        math.log(gamma / alpha) * (beta - alpha))
    magnitude = magnitude.round().clamp(max=beta)
    mapped = (torch.sign(outside_position) * magnitude).to(torch.long)

    indices = relative_position.clone().to(torch.long)
    indices[outside] = mapped
    return indices


@torch.no_grad()
def _build_product_buckets(height, width, ratio=1.9, skip=1):
    """Build the contextual Product-iRPE indices used by SAR-JEPA."""
    alpha = ratio
    beta = 2 * ratio
    gamma = 8 * ratio
    beta_int = int(beta)
    side = 2 * beta_int + 1

    rows = torch.arange(height).view(height, 1).repeat(1, width)
    cols = torch.arange(width).view(1, width).repeat(height, 1)
    positions = torch.stack((rows, cols), dim=2).reshape(-1, 2)
    differences = positions[:, None, :] - positions[None, :, :]
    row_indices = _piecewise_index(
        differences[:, :, 0], alpha, beta, gamma) + beta_int
    col_indices = _piecewise_index(
        differences[:, :, 1], alpha, beta, gamma) + beta_int
    buckets = row_indices * side + col_indices
    num_buckets = side**2

    if skip == 1:
        length = height * width
        with_cls = buckets.new_empty(length + 1, length + 1)
        with_cls[0] = num_buckets
        with_cls[:, 0] = num_buckets
        with_cls[1:, 1:] = buckets
        buckets = with_cls
    return buckets.contiguous(), num_buckets + skip


class ContextualIRPEKey(nn.Module):
    """Shared-head contextual iRPE attached to attention keys."""

    def __init__(self, head_dims, num_buckets):
        super().__init__()
        self.num_buckets = num_buckets
        self.head_dims = head_dims
        self.lookup_table_weight = nn.Parameter(
            torch.zeros(1, head_dims, num_buckets))

    def forward(self, query, relative_position_bucket):
        batch_size, _, length, head_dims = query.shape
        lookup = torch.matmul(
            query.transpose(0, 1).reshape(-1, batch_size * length,
                                          head_dims),
            self.lookup_table_weight)
        lookup = lookup.view(
            -1, batch_size, length, self.num_buckets).transpose(0, 1)

        offsets = torch.arange(
            0,
            length * self.num_buckets,
            self.num_buckets,
            dtype=relative_position_bucket.dtype,
            device=relative_position_bucket.device).view(-1, 1)
        flattened_indices = (relative_position_bucket + offsets).flatten()
        return lookup.flatten(2)[:, :, flattened_indices].view(
            batch_size, -1, length, length)


class SARJEPAImagePatchEmbed(nn.Module):
    """Checkpoint-compatible non-overlapping patch embedding."""

    def __init__(self, img_size, patch_size, in_channels, embed_dims):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = tuple(
            size // patch for size, patch in zip(self.img_size,
                                                 self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size)

    def forward(self, inputs):
        if inputs.shape[-2:] != self.img_size:
            raise ValueError(
                f'Expected input size {self.img_size}, got '
                f'{tuple(inputs.shape[-2:])}.')
        return self.proj(inputs).flatten(2).transpose(1, 2)


class SARJEPAMLP(nn.Module):
    """MLP retaining the official ``fc1`` and ``fc2`` state names."""

    def __init__(self, embed_dims, hidden_dims, drop_rate=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dims, embed_dims)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, inputs):
        inputs = self.drop(self.act(self.fc1(inputs)))
        return self.drop(self.fc2(inputs))


class SARJEPAAttention(nn.Module):
    """ViT attention with SAR-JEPA's key-side contextual Product-iRPE."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_buckets,
                 qkv_bias=True,
                 attn_drop_rate=0.,
                 drop_rate=0.):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads.')
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)
        self.rpe_k = ContextualIRPEKey(self.head_dims, num_buckets)

    def forward(self, inputs, relative_position_bucket):
        batch_size, num_tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size, num_tokens, 3, self.num_heads,
            self.head_dims).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention + self.rpe_k(
            query, relative_position_bucket)
        attention = self.attn_drop(attention.softmax(dim=-1))
        outputs = attention @ value
        outputs = outputs.transpose(1, 2).reshape(
            batch_size, num_tokens, channels)
        return self.proj_drop(self.proj(outputs))


class SARJEPABlock(nn.Module):
    """Checkpoint-compatible SAR-JEPA transformer block."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_buckets,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.attn = SARJEPAAttention(
            embed_dims,
            num_heads,
            num_buckets,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.mlp = SARJEPAMLP(
            embed_dims, int(embed_dims * mlp_ratio), drop_rate=drop_rate)

    def forward(self, inputs, relative_position_bucket):
        inputs = inputs + self.drop_path(
            self.attn(self.norm1(inputs), relative_position_bucket))
        return inputs + self.drop_path(self.mlp(self.norm2(inputs)))


class SARJEPANorm2d(nn.Module):
    """LayerNorm over channels of an NCHW feature map."""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = self.ln(inputs)
        return inputs.permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class SARJEPAVisionTransformer(BaseModule):
    """SAR-JEPA ViT-B/16 encoder adapted for semantic segmentation.

    The released checkpoint stores a complete masked predictive training
    model under ``checkpoint['model']`` and prefixes its tensors with
    ``module.``. Only the class token, one-channel patch embedding, twelve
    encoder blocks (including iRPE), and final norm are transferred, matching
    the official fine-tuning code. The prediction decoder, mask token and
    multi-scale gradient target modules are pre-training-only.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=1,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 out_indices=(3, 5, 7, 11),
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 with_cp=False,
                 collapse_repeated_input=True,
                 strict_encoder_load=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if patch_size != 16:
            raise ValueError('The SAR-JEPA dense FPN requires patch_size=16.')
        if len(out_indices) != 4:
            raise ValueError('Exactly four out_indices are required.')
        if tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError(
                'out_indices must be sorted and contain no duplicates.')

        self.img_size = to_2tuple(img_size)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.out_indices = tuple(out_indices)
        self.with_cp = with_cp
        self.collapse_repeated_input = collapse_repeated_input
        self.strict_encoder_load = strict_encoder_load

        self.patch_embed = SARJEPAImagePatchEmbed(
            self.img_size, patch_size, in_channels, embed_dims)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))

        relative_position_bucket, num_buckets = _build_product_buckets(
            *self.patch_embed.grid_size, ratio=1.9, skip=1)
        self.register_buffer(
            'relative_position_bucket',
            relative_position_bucket,
            persistent=False)

        drop_path_rates = torch.linspace(
            0, drop_path_rate, num_layers).tolist()
        self.blocks = ModuleList([
            SARJEPABlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_buckets=num_buckets,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[index])
            for index in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dims, eps=1e-6)

        # Simple feature pyramid used by ViT dense-prediction backbones.
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2),
            SARJEPANorm2d(embed_dims), nn.GELU(),
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)

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
            print_log('SAR-JEPA ViT-B initialized without pretraining.')
            return

        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        is_pretrained = init_cfg.get('type') == 'Pretrained'
        if not is_pretrained or 'checkpoint' not in init_cfg:
            raise ValueError(
                'SARJEPAVisionTransformer requires a Pretrained init_cfg '
                'with checkpoint.')

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
        encoder_prefixes = ('cls_token', 'patch_embed.', 'blocks.', 'norm.')
        required = {
            key for key in target
            if key == 'cls_token' or key.startswith(encoder_prefixes[1:])
        }
        loadable = OrderedDict()
        incompatible_source = []
        for key, value in source.items():
            if not (key == 'cls_token' or key.startswith(
                    encoder_prefixes[1:])):
                continue
            if key not in target or value.shape != target[key].shape:
                incompatible_source.append(key)
            else:
                loadable[key] = value
        missing = sorted(required.difference(loadable))

        if self.strict_encoder_load and (missing or incompatible_source):
            raise RuntimeError(
                'SAR-JEPA encoder checkpoint is not an exact match: '
                f'{len(missing)} missing keys and '
                f'{len(incompatible_source)} incompatible keys. '
                f'First missing keys: {missing[:5]}; first incompatible '
                f'keys: {incompatible_source[:5]}')

        self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SAR-JEPA encoder tensors from '
            f'{checkpoint_path}; ignored mask/predictor/gradient-target '
            'pre-training tensors. The dense feature pyramid remains '
            'randomly initialized.')

        del checkpoint, source, loadable
        gc.collect()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def _adapt_input_channels(self, inputs):
        if inputs.shape[1] == self.in_channels:
            return inputs
        if (self.collapse_repeated_input and self.in_channels == 1 and
                inputs.shape[1] == 3):
            # DDHR's grayscale loader replicates SAR into three channels.
            # Selecting one copy preserves the original values exactly;
            # averaging can introduce a small floating-point round-off.
            return inputs[:, :1]
        raise ValueError(
            f'Expected {self.in_channels} input channels, got '
            f'{inputs.shape[1]}.')

    def forward(self, inputs):
        inputs = self._adapt_input_channels(inputs)
        batch_size = inputs.shape[0]
        tokens = self.patch_embed(inputs)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        features = []
        for index, block in enumerate(self.blocks):
            if self.with_cp and self.training:
                tokens = checkpointing.checkpoint(
                    block,
                    tokens,
                    self.relative_position_bucket,
                    use_reentrant=False)
            else:
                tokens = block(tokens, self.relative_position_bucket)

            if index in self.out_indices:
                output_tokens = self.norm(tokens) if index == (
                    self.num_layers - 1) else tokens
                height, width = self.patch_embed.grid_size
                features.append(
                    output_tokens[:, 1:].transpose(1, 2).reshape(
                        batch_size, self.embed_dims, height, width))

        operations = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        return tuple(operation(feature)
                     for operation, feature in zip(operations, features))
