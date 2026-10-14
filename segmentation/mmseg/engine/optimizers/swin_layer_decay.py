"""Unified layer-wise learning-rate decay for Swin-B adapters."""

import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _stage_offset(depths, stage_index):
    """Number of blocks and PatchMerges preceding a Swin stage."""
    return sum(depths[:stage_index]) + stage_index


def _mmseg_swin_layer_id(name, depths, max_layer_id):
    if (name.startswith('backbone.patch_embed') or
            name == 'backbone.absolute_pos_embed'):
        return 0
    if name.startswith('backbone.stages.'):
        parts = name.split('.')
        stage_index = int(parts[2])
        offset = _stage_offset(depths, stage_index)
        if len(parts) > 4 and parts[3] == 'blocks':
            return offset + int(parts[4]) + 1
        if len(parts) > 3 and parts[3] == 'downsample':
            return offset + depths[stage_index] + 1
    if name.startswith('backbone.norm'):
        suffix = name[len('backbone.norm'):].split('.')[0]
        if suffix.isdigit():
            stage_index = int(suffix)
            return _stage_offset(depths, stage_index) + depths[stage_index]
    return max_layer_id


def _torchvision_swin_layer_id(name, depths, max_layer_id):
    patch_prefix = 'backbone.backbone.features.0.'
    feature_prefix = 'backbone.backbone.features.'
    if name.startswith(patch_prefix):
        return 0
    if name.startswith(feature_prefix):
        parts = name.split('.')
        feature_index = int(parts[3])
        if feature_index in (1, 3, 5, 7):
            stage_index = (feature_index - 1) // 2
            offset = _stage_offset(depths, stage_index)
            block_index = int(parts[4])
            return offset + block_index + 1
        if feature_index in (2, 4, 6):
            stage_index = feature_index // 2 - 1
            offset = _stage_offset(depths, stage_index)
            return offset + depths[stage_index] + 1
    return max_layer_id


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SwinLayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Count every Swin block and PatchMerge as one sequential layer.

    This supports both MMSeg's Swin implementation used by GFM and the
    torchvision Swin-V2 wrapper used by SatlasPretrain. The final Swin block,
    output norms, segmentation neck and decoder use the unscaled base LR.
    """

    def add_params(self, params: List[dict], module: nn.Module, **kwargs):
        logger = MMLogger.get_current_instance()
        expected_blocks = self.paramwise_cfg['num_layers']
        decay_rate = self.paramwise_cfg['decay_rate']
        backbone = module.backbone

        if hasattr(backbone, 'stages'):
            depths = tuple(len(stage.blocks) for stage in backbone.stages)
            layer_id_fn = _mmseg_swin_layer_id
            implementation = 'MMSeg Swin'
        elif (hasattr(backbone, 'backbone') and
              hasattr(backbone.backbone, 'features')):
            features = backbone.backbone.features
            depths = tuple(len(features[index]) for index in (1, 3, 5, 7))
            layer_id_fn = _torchvision_swin_layer_id
            implementation = 'torchvision Swin-V2'
        else:
            raise TypeError(
                'SwinLayerDecayOptimizerConstructor only supports the GFM '
                'and Satlas Swin backbone layouts.')

        if sum(depths) != expected_blocks:
            raise ValueError(
                f'Expected {expected_blocks} Swin blocks, found {sum(depths)} '
                f'with stage depths {depths}.')
        max_layer_id = sum(depths) + len(depths) - 1
        parameter_groups = {}

        no_decay_names = (
            'absolute_pos_embed', 'relative_position_bias_table',
            'relative_position_index', 'logit_scale')
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            no_decay = (parameter.ndim == 1 or name.endswith('.bias') or
                        any(item in name for item in no_decay_names))
            decay_tag = 'no_decay' if no_decay else 'decay'
            weight_decay = 0. if no_decay else self.base_wd
            layer_id = layer_id_fn(name, depths, max_layer_id)
            group_name = f'layer_{layer_id}_{decay_tag}'
            if group_name not in parameter_groups:
                lr_scale = decay_rate**(max_layer_id - layer_id)
                parameter_groups[group_name] = dict(
                    params=[],
                    param_names=[],
                    lr=self.base_lr * lr_scale,
                    lr_scale=lr_scale,
                    weight_decay=weight_decay,
                    group_name=group_name)
            parameter_groups[group_name]['params'].append(parameter)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            summary = {
                key: {
                    'lr_scale': value['lr_scale'],
                    'lr': value['lr'],
                    'weight_decay': value['weight_decay'],
                    'num_params': len(value['param_names'])
                }
                for key, value in parameter_groups.items()
            }
            logger.info(
                f'{implementation} depths={depths}, sequential layers='
                f'{max_layer_id}, parameter groups: ' +
                json.dumps(summary, indent=2))
        params.extend(parameter_groups.values())
