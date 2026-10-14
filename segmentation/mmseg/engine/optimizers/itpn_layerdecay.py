# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import List

import torch.nn as nn
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed", "backbone.absolute_pos_embed"):
        return 0
    elif var_name.startswith("backbone.patch_embed"):
        return 0
    elif var_name.startswith("backbone.blocks"):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return num_max_layer - 1

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class itpn_LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):

    def add_params(self, params: List[dict], module: nn.Module, **kwargs) -> None:
        logger = MMLogger.get_current_instance()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 3
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                    len(param.shape) == 1 or name.endswith(".bias") or
                    name.endswith("_token") or name.endswith("pos_embed")
            ):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
                
            # if 'layer_wise' in decay_type:
            #     logger.info('Layer-wise decay constructing')
            layer_id = get_num_layer_for_vit(name, num_layers)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            logger.info(f'set param {name} as id {layer_id}')
            
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())
