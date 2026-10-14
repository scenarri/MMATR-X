"""SARMAE ViT backbone adapter for MMSegmentation."""

import gc
from collections import OrderedDict

import torch.nn as nn
from mmengine.logging import print_log

from mmseg.models.backbones.selective_mae_vit import (
    SelectiveMAEVisionTransformer)
from mmseg.registry import MODELS


@MODELS.register_module()
class SARMAEVisionTransformer(SelectiveMAEVisionTransformer):
    """SARMAE ViT-B encoder following its official segmentation port.

    SARMAE and SelectiveMAE use the same downstream no-class-token ViT and
    Simple FPN structure, so the numerically verified implementation is
    shared. This adapter specializes checkpoint loading for SARMAE's complete
    SAR--optical training snapshot, whose transferable encoder is stored
    below ``sar_encoder.``.
    """

    def __init__(self, strict_encoder_load=True, **kwargs):
        self.strict_encoder_load = strict_encoder_load
        super().__init__(**kwargs)

    def init_weights(self):
        self.apply(self._initialize_module)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.init_cfg is None:
            print_log('SARMAE ViT-B initialized without pretraining.')
            return

        init_cfg = self.init_cfg[0] if isinstance(
            self.init_cfg, list) else self.init_cfg
        if (init_cfg.get('type') != 'Pretrained' or
                'checkpoint' not in init_cfg):
            raise ValueError(
                'SARMAEVisionTransformer requires a Pretrained init_cfg '
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

        sar_prefix = 'sar_encoder.'
        sar_source = OrderedDict(
            (key[len(sar_prefix):], value)
            for key, value in source.items()
            if key.startswith(sar_prefix))
        if sar_source:
            source = sar_source

        if 'pos_embed' not in source:
            raise RuntimeError(
                'No SARMAE encoder position embedding was found in the '
                'checkpoint.')
        source['pos_embed'] = self._resize_position_embedding(
            source['pos_embed'])

        target = self.state_dict()
        encoder_prefixes = ('pos_embed', 'patch_embed.', 'blocks.')
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
                'SARMAE encoder checkpoint is not an exact match: '
                f'{len(missing)} missing and {len(incompatible_source)} '
                f'incompatible keys. First missing: {missing[:5]}; first '
                f'incompatible: {incompatible_source[:5]}')

        msg = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} SARMAE encoder tensors from '
            f'{checkpoint_path}; ignored the class token, final norm, MAE '
            'decoder, optical encoder and alignment head. The dense feature '
            'pyramid remains newly initialized.')
        print_log(msg)

        del checkpoint, source, sar_source, loadable
        gc.collect()
