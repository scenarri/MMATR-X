"""Layer-wise learning-rate decay for the SARMAE ViT encoder."""

import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _get_layer_id(name, num_layers):
    if name == 'backbone.pos_embed':
        return 0
    if name.startswith('backbone.patch_embed'):
        return 0
    if name.startswith('backbone.blocks'):
        return int(name.split('.')[2]) + 1
    # The last transformer block, pyramid and decoder use the base LR.
    return num_layers


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SARMAELayerDecayOptimizerConstructor(
        DefaultOptimWrapperConstructor):
    """Apply the project's unified 12-layer ViT decay grouping."""

    def add_params(self, params: List[dict], module: nn.Module, **kwargs):
        logger = MMLogger.get_current_instance()
        parameter_groups = {}
        num_layers = self.paramwise_cfg['num_layers']
        decay_rate = self.paramwise_cfg['decay_rate']
        max_layer_id = num_layers

        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            no_decay = (parameter.ndim == 1 or name.endswith('.bias') or
                        'pos_embed' in name)
            decay_tag = 'no_decay' if no_decay else 'decay'
            weight_decay = 0. if no_decay else self.base_wd
            layer_id = _get_layer_id(name, num_layers)
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
            logger.info('SARMAE parameter groups: ' +
                        json.dumps(summary, indent=2))
        params.extend(parameter_groups.values())
