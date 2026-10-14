"""SatlasPretrain Swin-V2 backbone adapter for MMSegmentation."""

import gc
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import (
    PatchMergingV2, SwinTransformer, SwinTransformerBlockV2)
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS


@MODELS.register_module()
class SatlasSwinTransformerV2(BaseModule):
    """Official SatlasPretrain Swin-V2 backbone for dense prediction.

    The released ``Aerial_SwinB_SI`` checkpoint is a complete multi-task
    checkpoint. Its Swin-V2-B weights are stored below the
    ``backbone.backbone.`` prefix. SatlasPretrain constructs the backbone with
    ``torchvision.models.swin_v2_b`` and returns the outputs of feature stages
    1, 3, 5 and 7. This adapter retains that implementation and loading
    convention while exposing the four maps directly to MMSegmentation.

    The torchvision classification norm and head are retained so that all 449
    backbone tensors load exactly, but frozen because the official dense
    feature forward does not execute them.

    Args:
        in_channels (int): Number of image channels. Defaults to 3.
        out_indices (tuple[int]): Swin stages to return. Defaults to all four.
        drop_path_rate (float): Maximum stochastic-depth probability.
            Defaults to 0.1.
        strict_load (bool): Require an exact backbone checkpoint match.
            Defaults to True.
        init_cfg (dict, optional): MMEngine initialization configuration.
    """

    def __init__(self,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.1,
                 strict_load=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError(
                'out_indices must be sorted and contain no duplicates.')
        if any(index not in range(4) for index in out_indices):
            raise ValueError('out_indices must only contain 0, 1, 2 and 3.')

        self.in_channels = in_channels
        self.out_indices = tuple(out_indices)
        self.drop_path_rate = drop_path_rate
        self.strict_load = strict_load
        self.out_channels = (128, 256, 512, 1024)

        # ``torchvision.models.swin_v2_b`` hard-codes stochastic depth to 0.5,
        # so instantiate its base class with the exact same architecture while
        # exposing that one value to the downstream fine-tuning config.
        self.backbone = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[8, 8],
            stochastic_depth_prob=drop_path_rate,
            block=SwinTransformerBlockV2,
            downsample_layer=PatchMergingV2)
        if in_channels != 3:
            patch_embed = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                patch_embed.out_channels,
                kernel_size=patch_embed.kernel_size,
                stride=patch_embed.stride,
                padding=patch_embed.padding,
                bias=patch_embed.bias is not None)

        # These layers belong to the ImageNet classification path and are not
        # called by SatlasPretrain's multi-scale feature forward.
        self.backbone.norm.requires_grad_(False)
        self.backbone.head.requires_grad_(False)

    @staticmethod
    def _load_checkpoint(checkpoint):
        if isinstance(checkpoint, str) and osp.isfile(checkpoint):
            try:
                return torch.load(checkpoint, map_location='cpu', mmap=True)
            except (RuntimeError, TypeError):
                return torch.load(checkpoint, map_location='cpu')
        return CheckpointLoader.load_checkpoint(
            checkpoint, logger=None, map_location='cpu')

    @staticmethod
    def _extract_backbone(state_dict):
        """Select Satlas backbone tensors and normalize their prefix."""
        extracted = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]

            if key.startswith('backbone.backbone.'):
                # Full Satlas model -> this adapter.
                key = key[len('backbone.'):]
            elif key.startswith('backbone.'):
                # Already extracted official SwinBackbone state dict.
                pass
            elif key.startswith(('features.', 'norm.', 'head.')):
                # Raw torchvision Swin-V2 state dict.
                key = f'backbone.{key}'
            else:
                continue
            extracted[key] = value
        return extracted

    def init_weights(self):
        if self.init_cfg is None:
            print_log('Satlas Swin-V2-B initialized without pretraining.')
            return

        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        is_pretrained = init_cfg.get('type') == 'Pretrained'
        if not is_pretrained or 'checkpoint' not in init_cfg:
            raise ValueError(
                'SatlasSwinTransformerV2 requires a Pretrained init_cfg '
                'with checkpoint.')

        checkpoint_path = init_cfg['checkpoint']
        checkpoint = self._load_checkpoint(checkpoint_path)
        if 'state_dict' in checkpoint:
            source = checkpoint['state_dict']
        elif 'model' in checkpoint:
            source = checkpoint['model']
        else:
            source = checkpoint

        source = self._extract_backbone(source)
        target = self.state_dict()
        loadable = OrderedDict()
        incompatible_source = []
        for key, value in source.items():
            if key not in target or value.shape != target[key].shape:
                incompatible_source.append(key)
            else:
                loadable[key] = value
        missing = [key for key in target if key not in loadable]

        if self.strict_load and (missing or incompatible_source):
            raise RuntimeError(
                'Satlas backbone checkpoint is not an exact match: '
                f'{len(missing)} missing target keys and '
                f'{len(incompatible_source)} incompatible source keys. '
                f'First missing keys: {missing[:5]}; first incompatible '
                f'keys: {incompatible_source[:5]}')

        incompatible = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SatlasPretrain Swin-V2-B tensors from '
            f'{checkpoint_path}; ignored the checkpoint FPN and task heads.')
        print_log(str(incompatible))

        del checkpoint, source, loadable
        gc.collect()

    def forward(self, inputs):
        outputs = []
        stage_index = 0
        for feature_index, layer in enumerate(self.backbone.features):
            inputs = layer(inputs)
            if feature_index not in (1, 3, 5, 7):
                continue
            if stage_index in self.out_indices:
                outputs.append(inputs.permute(0, 3, 1, 2))
            stage_index += 1
        return tuple(outputs)
