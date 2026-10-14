"""Shared and independent dual-modal iTPN feature extractors."""

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmseg.models.backbones.iTPN_pixel_dualembed import (
    PatchEmbed, PatchMerge, iTPN_pixel_dualembed)
from mmseg.registry import MODELS


def _validate_modal_inputs(rgb: Tensor, sar: Tensor) -> None:
    if rgb.ndim != 4 or sar.ndim != 4:
        raise ValueError('RGB and SAR inputs must both be BCHW tensors')
    if rgb.shape[0] != sar.shape[0] or rgb.shape[2:] != sar.shape[2:]:
        raise ValueError(
            'RGB and SAR must have equal batch and spatial shapes, but got '
            f'{tuple(rgb.shape)} and {tuple(sar.shape)}')


def _fuse_pyramids(rgb_features: Sequence[Tensor],
                   sar_features: Sequence[Tensor]) -> Tuple[Tensor, ...]:
    if len(rgb_features) != len(sar_features):
        raise ValueError('RGB and SAR backbones returned different levels')
    fused = []
    for level, (rgb, sar) in enumerate(zip(rgb_features, sar_features)):
        if rgb.shape[0] != sar.shape[0] or rgb.shape[2:] != sar.shape[2:]:
            raise ValueError(
                f'Feature shapes do not match at level {level}: '
                f'{tuple(rgb.shape)} and {tuple(sar.shape)}')
        fused.append(torch.cat((rgb, sar), dim=1))
    return tuple(fused)


@MODELS.register_module()
class SharedDualModalITPN(iTPN_pixel_dualembed):
    """iTPN with modality-specific patch embeddings and shared later layers.

    RGB and SAR tokens are concatenated along the batch dimension immediately
    after patch embedding.  Every subsequent block and the FPN are shared.
    The two output halves are finally concatenated along channel dimension.
    """

    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 rgb_in_chans=3,
                 sar_in_chans=3,
                 embed_dim=512,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen=False,
                 **kwargs) -> None:
        kwargs.pop('modality', None)
        kwargs.pop('in_chans', None)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=rgb_in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            modality='RGB',
            frozen=frozen,
            **kwargs)

        self.rgb_in_chans = rgb_in_chans
        self.sar_in_chans = sar_in_chans
        self.patch_embed_sar = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=sar_in_chans,
            embed_dim=embed_dim // 4,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_sar.apply(self._init_weights)
        # The parent may have frozen itself before the SAR embedding existed.
        self._freeze()

    def _forward_from_embeddings(self, x: Tensor, hp: int,
                                 wp: int) -> Tuple[Tensor, ...]:
        batch_size = x.shape[0]
        lateral_features = []
        for block in self.blocks[:-self.num_main_blocks]:
            if isinstance(block, PatchMerge):
                lateral_features.append(x)
            x = (checkpoint.checkpoint(block, x)
                 if self.use_checkpoint else block(x))

        x = x[..., 0, 0, :]
        if self.ape:
            x = x + self.interpolate_pos_encoding(x, hp, wp)
        x = self.pos_drop(x)

        rpe_index = (self.relative_position_index.view(-1)
                     if self.rpe else None)
        for block in self.blocks[-self.num_main_blocks:]:
            x = (checkpoint.checkpoint(block, x, rpe_index, None)
                 if self.use_checkpoint else block(x, rpe_index, None))

        if self.num_outs == -1:
            return (x, )

        x = x[..., None, None, :]
        outputs = ([x] if self.align_dim_16tofpn is None else
                   [self.align_dim_16tofpn(x)])
        if self.num_outs >= 2:
            x = self.block_16to8(
                self.split_16to8(x) +
                self.align_dim_16to8(lateral_features[1]))
            outputs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(
                self.split_8to4(x) +
                self.align_dim_8to4(lateral_features[0]))
            outputs.append(x)

        for index, output in enumerate(outputs):
            outputs[index] = self.fpn_modules[index](output)

        outputs = [
            output.reshape(batch_size, hp, wp, *output.shape[-3:])
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch_size, -1, hp * output.shape[-3],
                     wp * output.shape[-2]).contiguous()
            for output in outputs
        ]
        if self.num_outs >= 4:
            outputs.insert(0, F.max_pool2d(outputs[0], 2, 2))
        if self.num_outs >= 5:
            outputs.insert(0, F.max_pool2d(outputs[0], 2, 2))
        return tuple(reversed(outputs))

    def forward(self, rgb: Tensor, sar: Tensor) -> Tuple[Tensor, ...]:
        _validate_modal_inputs(rgb, sar)
        if rgb.size(1) != self.rgb_in_chans:
            raise ValueError(f'Expected {self.rgb_in_chans} RGB channels')
        if sar.size(1) != self.sar_in_chans:
            raise ValueError(f'Expected {self.sar_in_chans} SAR channels')

        height, width = rgb.shape[2:]
        patch_h, patch_w = self.patch_embed_rgb.patch_size
        if height % patch_h or width % patch_w:
            raise ValueError('Input size must be divisible by patch_size')
        hp, wp = height // patch_h, width // patch_w

        rgb_tokens = self.patch_embed_rgb(rgb)
        sar_tokens = self.patch_embed_sar(sar)
        joint_tokens = torch.cat((rgb_tokens, sar_tokens), dim=0)
        joint_features = self._forward_from_embeddings(joint_tokens, hp, wp)

        batch_size = rgb.size(0)
        rgb_features = [item[:batch_size] for item in joint_features]
        sar_features = [item[batch_size:] for item in joint_features]
        return _fuse_pyramids(rgb_features, sar_features)


@MODELS.register_module()
class IndependentDualModalITPN(BaseModule):
    """Two independent backbones followed by channel-wise pyramid fusion."""

    def __init__(self,
                 rgb_backbone: dict,
                 sar_backbone: dict,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.rgb_backbone = MODELS.build(rgb_backbone)
        self.sar_backbone = MODELS.build(sar_backbone)

    def forward(self, rgb: Tensor, sar: Tensor) -> Tuple[Tensor, ...]:
        _validate_modal_inputs(rgb, sar)
        rgb_features = self.rgb_backbone(rgb)
        sar_features = self.sar_backbone(sar)
        return _fuse_pyramids(rgb_features, sar_features)

    def get_num_layers(self) -> int:
        return max(self.rgb_backbone.get_num_layers(),
                   self.sar_backbone.get_num_layers())
