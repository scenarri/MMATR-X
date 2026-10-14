"""ViT backbone adapter for the official SelectiveMAE checkpoint."""

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
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS


class PatchEmbed(nn.Module):
    """Non-overlapping patch embedding with official timm-style names."""

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


class Attention(nn.Module):
    """Standard ViT attention used by the SelectiveMAE encoder."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 qkv_bias=True,
                 attn_drop_rate=0.,
                 drop_rate=0.,
                 use_sdpa=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims**-0.5
        self.use_sdpa = use_sdpa and hasattr(
            F, 'scaled_dot_product_attention')
        self.attn_drop_rate = attn_drop_rate
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, inputs):
        batch_size, num_tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch_size, num_tokens, 3, self.num_heads,
            self.head_dims).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attn_drop_rate if self.training else 0.)
        else:
            attention = (query * self.scale) @ key.transpose(-2, -1)
            attention = attention.softmax(dim=-1)
            attention = F.dropout(
                attention, p=self.attn_drop_rate, training=self.training)
            output = attention @ value

        output = output.transpose(1, 2).reshape(
            batch_size, num_tokens, channels)
        output = self.proj(output)
        return self.proj_drop(output)


class MLP(nn.Module):
    """Two-layer MLP with checkpoint-compatible parameter names."""

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


class TransformerBlock(nn.Module):
    """Pre-norm ViT block matching the SelectiveMAE online encoder."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_sdpa=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.attn = Attention(
            embed_dims,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
            use_sdpa=use_sdpa)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dims, eps=1e-6)
        self.mlp = MLP(
            embed_dims, int(embed_dims * mlp_ratio), drop_rate=drop_rate)

    def forward(self, inputs):
        inputs = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        return inputs + self.drop_path(self.mlp(self.norm2(inputs)))


class Norm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = self.ln(inputs)
        return inputs.permute(0, 3, 1, 2).contiguous()


@MODELS.register_module()
class SelectiveMAEVisionTransformer(BaseModule):
    """SelectiveMAE ViT encoder adapted for MMSegmentation 1.x.

    Selective masking, HOG guidance and reconstruction decoders are used only
    during pre-training. The official semantic-segmentation code retains the
    standard ViT encoder, removes the class token, and constructs its four
    output scales with an internal feature pyramid. This implementation follows
    that downstream architecture without depending on the repository's timm
    version.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
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
                 use_sdpa=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if patch_size != 16:
            raise ValueError('The SelectiveMAE FPN requires patch_size=16.')
        if len(out_indices) != 4:
            raise ValueError('Exactly four out_indices are required.')

        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        self.out_indices = tuple(out_indices)
        self.with_cp = with_cp
        self.patch_embed = PatchEmbed(self.img_size, patch_size, in_channels,
                                      embed_dims)

        # The released checkpoint stores a 14x14 sin-cos grid plus one class
        # position. Fine-tuning discards the class position and learns this
        # image-only parameter after interpolation to the target patch grid.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dims))
        self.pos_drop = nn.Dropout(drop_rate)

        drop_path_rates = torch.linspace(0, drop_path_rate,
                                         num_layers).tolist()
        self.blocks = ModuleList([
            TransformerBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[index],
                use_sdpa=use_sdpa) for index in range(num_layers)
        ])

        # Equivalent to Feature2Pyramid rescales [4, 2, 1, 0.5], with the
        # LayerNorm2d used by the official SelectiveMAE segmentation port.
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2),
            Norm2d(embed_dims), nn.GELU(),
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _load_checkpoint(checkpoint):
        # The release also contains optimizer/scaler and decoder states and is
        # about 1.56 GB. mmap prevents those unused entries being copied into
        # host RAM during backbone initialization.
        if isinstance(checkpoint, str) and osp.isfile(checkpoint):
            try:
                return torch.load(checkpoint, map_location='cpu', mmap=True)
            except (RuntimeError, TypeError):
                return torch.load(checkpoint, map_location='cpu')
        return CheckpointLoader.load_checkpoint(
            checkpoint, logger=None, map_location='cpu')

    def _resize_position_embedding(self, position_embedding):
        source_tokens = position_embedding.shape[1]
        source_size = int(math.sqrt(source_tokens))
        if source_size * source_size != source_tokens:
            source_size = int(math.sqrt(source_tokens - 1))
            if source_size * source_size != source_tokens - 1:
                raise ValueError(
                    'Cannot infer the SelectiveMAE position grid from '
                    f'{tuple(position_embedding.shape)}.')
            position_embedding = position_embedding[:, 1:]

        target_h, target_w = self.patch_embed.grid_size
        if (source_size, source_size) != (target_h, target_w):
            position_embedding = position_embedding.reshape(
                1, source_size, source_size,
                position_embedding.shape[-1]).permute(0, 3, 1, 2)
            position_embedding = F.interpolate(
                position_embedding,
                size=(target_h, target_w),
                mode='bicubic',
                align_corners=False)
            position_embedding = position_embedding.permute(
                0, 2, 3, 1).flatten(1, 2)
        return position_embedding

    def init_weights(self):
        self.apply(self._initialize_module)
        trunc_normal_(self.pos_embed, std=.02)
        if self.init_cfg is None:
            print_log('SelectiveMAE ViT-B initialized without pretraining.')
            return

        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        if init_cfg.get('type') != 'Pretrained' or 'checkpoint' not in init_cfg:
            raise ValueError(
                'SelectiveMAEVisionTransformer requires a Pretrained '
                'init_cfg with checkpoint.')

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
        if 'pos_embed' in source:
            source['pos_embed'] = self._resize_position_embedding(
                source['pos_embed'])

        target = self.state_dict()
        loadable = OrderedDict()
        skipped = []
        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                loadable[key] = value
            else:
                skipped.append(key)

        incompatible = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SelectiveMAE encoder tensors from '
            f'{checkpoint_path}; ignored {len(skipped)} pre-training-only or '
            'incompatible tensors.')
        print_log(str(incompatible))

        del checkpoint, source, loadable
        gc.collect()

    def get_num_layers(self):
        return self.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        features = []
        tokens = self.patch_embed(inputs)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for index, block in enumerate(self.blocks):
            if self.with_cp and self.training:
                tokens = checkpointing.checkpoint(
                    block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if index in self.out_indices:
                height, width = self.patch_embed.grid_size
                features.append(
                    tokens.transpose(1, 2).reshape(
                        batch_size, self.embed_dims, height, width))

        operations = (self.fpn1, self.fpn2, self.fpn3, self.fpn4)
        return tuple(operation(feature)
                     for operation, feature in zip(operations, features))

