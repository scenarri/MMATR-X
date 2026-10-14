"""Layer-wise learning-rate decay for shared/independent dual-modal iTPN."""

import json
from typing import List

from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor
from torch import nn

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _local_backbone_name(parameter_name: str) -> str:
    prefixes = (
        'backbone.rgb_backbone.',
        'backbone.sar_backbone.',
        'backbone.',
    )
    for prefix in prefixes:
        if parameter_name.startswith(prefix):
            return parameter_name[len(prefix):]
    return ''


def _get_layer_id(parameter_name: str, max_layer_id: int) -> int:
    local_name = _local_backbone_name(parameter_name)
    if not local_name:
        return max_layer_id - 1
    if local_name in {
            'cls_token', 'mask_token', 'pos_embed', 'absolute_pos_embed'
    } or local_name.startswith('patch_embed'):
        return 0
    if local_name.startswith('blocks.'):
        return int(local_name.split('.')[1]) + 1
    return max_layer_id - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class DualModalITPNLayerDecayOptimizerConstructor(
        DefaultOptimWrapperConstructor):
    """Assign the same layer decay to both branches of a dual iTPN."""

    def add_params(self, params: List[dict], module: nn.Module,
                   **kwargs) -> None:
        logger = MMLogger.get_current_instance()
        num_layers = self.paramwise_cfg['num_layers'] + 3
        decay_rate = self.paramwise_cfg['decay_rate']
        parameter_groups = {}

        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            no_decay = (parameter.ndim == 1 or name.endswith('.bias') or
                        name.endswith('_token') or name.endswith('pos_embed') or
                        'relative_position_bias_table' in name)
            decay_tag = 'no_decay' if no_decay else 'decay'
            weight_decay = 0.0 if no_decay else self.base_wd
            layer_id = _get_layer_id(name, num_layers)
            group_name = f'layer_{layer_id}_{decay_tag}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)
                parameter_groups[group_name] = dict(
                    weight_decay=weight_decay,
                    params=[],
                    param_names=[],
                    lr_scale=scale,
                    group_name=group_name,
                    lr=scale * self.base_lr)
            parameter_groups[group_name]['params'].append(parameter)
            parameter_groups[group_name]['param_names'].append(name)

        rank, _ = get_dist_info()
        if rank == 0:
            display = {
                key: {
                    'param_names': value['param_names'],
                    'lr_scale': value['lr_scale'],
                    'lr': value['lr'],
                    'weight_decay': value['weight_decay'],
                }
                for key, value in parameter_groups.items()
            }
            logger.info(
                'Dual-modal iTPN parameter groups = '
                f'{json.dumps(display, indent=2)}')
        params.extend(parameter_groups.values())
