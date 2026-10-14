"""Layer-wise learning-rate decay for SARATR-X HiViT."""

import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def _get_layer_id(name, num_backbone_layers):
    """Use the same sequential ``blocks`` indexing as the iTPN recipe."""
    if not name.startswith('backbone.'):
        return num_backbone_layers
    internal_name = name[len('backbone.'):]
    if (internal_name == 'absolute_pos_embed' or
            internal_name.startswith('patch_embed')):
        return 0
    if internal_name.startswith('blocks.'):
        block_index = int(internal_name.split('.')[1])
        return block_index + 1
    return num_backbone_layers


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class SARATRXLayerDecayOptimizerConstructor(
        DefaultOptimWrapperConstructor):
    """Apply the project's unified sequential HiViT decay assignment.

    The two shallow HiViT stages contain MLP-only residual blocks, and the
    two PatchMerge modules are trainable depth transitions. Consequently all
    entries in ``backbone.blocks`` advance the layer id. The last backbone
    block, pyramid and decoder share the unscaled base learning rate, exactly
    as in ``itpn_LayerDecayOptimizerConstructor``.
    """

    def add_params(self, params: List[dict], module: nn.Module, **kwargs):
        logger = MMLogger.get_current_instance()
        parameter_groups = {}
        num_residual_layers = self.paramwise_cfg['num_layers']
        decay_rate = self.paramwise_cfg['decay_rate']
        num_backbone_layers = len(module.backbone.blocks)
        residual_layers = sum(
            getattr(module.backbone, 'depths', ())) + sum(
                getattr(module.backbone, 'depths', ())[:-1])
        if residual_layers != num_residual_layers:
            raise ValueError(
                f'Expected {num_residual_layers} SARATR-X residual blocks, '
                f'found {residual_layers}.')
        if num_backbone_layers != num_residual_layers + 2:
            raise ValueError(
                'SARATR-X should contain two additional PatchMerge layers; '
                f'got {num_backbone_layers} sequential backbone layers.')

        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            no_decay = (
                parameter.ndim == 1 or name.endswith('.bias') or
                'absolute_pos_embed' in name or
                'relative_position_bias_table' in name)
            decay_tag = 'no_decay' if no_decay else 'decay'
            weight_decay = 0. if no_decay else self.base_wd
            layer_id = _get_layer_id(name, num_backbone_layers)
            group_name = f'layer_{layer_id}_{decay_tag}'
            if group_name not in parameter_groups:
                lr_scale = decay_rate**(
                    num_backbone_layers - layer_id)
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
            logger.info('SARATR-X parameter groups: ' +
                        json.dumps(summary, indent=2))
        params.extend(parameter_groups.values())
