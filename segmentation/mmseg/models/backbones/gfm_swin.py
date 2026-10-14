"""Swin backbone adapter for the official GFM pre-training checkpoint."""

import gc
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS
from .swin import SwinTransformer


def _correct_unfold_reduction_order(weight):
    """Convert the official Swin patch-merging order to MMCV's Unfold."""
    out_channels, in_channels = weight.shape
    weight = weight.reshape(out_channels, 4, in_channels // 4)
    return (weight[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(
        out_channels, in_channels))


def _correct_unfold_norm_order(parameter):
    """Convert an official Swin patch-merging norm vector."""
    channels = parameter.shape[0]
    parameter = parameter.reshape(4, channels // 4)
    return parameter[[0, 2, 1, 3], :].transpose(0, 1).reshape(channels)


def _convert_gfm_swin(state_dict, final_norm_name):
    """Convert official GFM/SimMIM Swin names to MMSeg 1.x names."""
    converted = OrderedDict()
    for key, value in state_dict.items():
        # These entries are either pre-training-only or deterministic buffers.
        if (key == 'mask_token' or key.startswith('head.') or
                key.endswith('attn_mask') or
                key.endswith('relative_position_index')):
            continue

        if key in ('norm.weight', 'norm.bias'):
            new_key = key.replace('norm', final_norm_name, 1)
            converted[new_key] = value
            continue

        new_key = key
        new_value = value
        if key.startswith('layers.'):
            if '.attn.' in new_key:
                new_key = new_key.replace('.attn.', '.attn.w_msa.', 1)
            elif '.mlp.fc1.' in new_key:
                new_key = new_key.replace(
                    '.mlp.fc1.', '.ffn.layers.0.0.', 1)
            elif '.mlp.fc2.' in new_key:
                new_key = new_key.replace('.mlp.fc2.', '.ffn.layers.1.', 1)
            elif '.mlp.' in new_key:
                new_key = new_key.replace('.mlp.', '.ffn.', 1)

            if '.downsample.reduction.weight' in key:
                new_value = _correct_unfold_reduction_order(value)
            elif '.downsample.norm.' in key:
                new_value = _correct_unfold_norm_order(value)
            new_key = new_key.replace('layers.', 'stages.', 1)
        elif key.startswith('patch_embed.proj.'):
            new_key = key.replace(
                'patch_embed.proj.', 'patch_embed.projection.', 1)

        converted[new_key] = new_value
    return converted


@MODELS.register_module()
class GFMSwinTransformer(SwinTransformer):
    """MMSeg Swin with direct loading of the official ``gfm.pth``.

    The released checkpoint is a complete continual-pretraining snapshot. Its
    ``model`` entry contains ``encoder``, ``teacher``, ``decoder`` and
    ``projector`` branches. The official fine-tuning code selects the online
    ``encoder`` branch. This adapter makes the same selection and converts its
    standard Swin names and patch-merging tensor order to MMSeg 1.x.

    Args:
        gfm_branch (str): Checkpoint branch to load. Keep ``encoder`` for the
            official fine-tuning behavior. Defaults to ``encoder``.
    """

    def __init__(self, gfm_branch='encoder', **kwargs):
        self.gfm_branch = gfm_branch
        super().__init__(**kwargs)

    @staticmethod
    def _load_checkpoint(checkpoint):
        # gfm.pth also stores optimizer/AMP states and is about 1.4 GB. mmap
        # avoids materializing those unused entries in host memory.
        if isinstance(checkpoint, str) and osp.isfile(checkpoint):
            try:
                return torch.load(checkpoint, map_location='cpu', mmap=True)
            except (RuntimeError, TypeError):
                return torch.load(checkpoint, map_location='cpu')
        return CheckpointLoader.load_checkpoint(
            checkpoint, logger=None, map_location='cpu')

    def init_weights(self):
        if self.init_cfg is None:
            return super().init_weights()

        if 'checkpoint' not in self.init_cfg:
            raise ValueError('GFMSwinTransformer init_cfg needs checkpoint')

        checkpoint_path = self.init_cfg['checkpoint']
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
        branch_prefix = f'{self.gfm_branch}.'
        if any(key.startswith(branch_prefix) for key in source):
            source = OrderedDict(
                (key[len(branch_prefix):], value)
                for key, value in source.items()
                if key.startswith(branch_prefix))
        elif any(key.startswith('encoder.') for key in source):
            raise KeyError(
                f'Branch {self.gfm_branch!r} is absent from {checkpoint_path}')

        # Also accept an already extracted backbone checkpoint.
        source = OrderedDict(
            (key[9:] if key.startswith('backbone.') else key, value)
            for key, value in source.items())

        final_stage = max(self.out_indices)
        source = _convert_gfm_swin(source, f'norm{final_stage}')
        target = self.state_dict()
        loadable = OrderedDict()
        skipped = []

        for key, value in source.items():
            if key not in target:
                skipped.append(key)
                continue

            if 'relative_position_bias_table' in key:
                current = target[key]
                src_length, src_heads = value.shape
                dst_length, dst_heads = current.shape
                if src_heads == dst_heads and src_length != dst_length:
                    src_size = int(src_length**0.5)
                    dst_size = int(dst_length**0.5)
                    value = F.interpolate(
                        value.permute(1, 0).reshape(
                            1, src_heads, src_size, src_size),
                        size=(dst_size, dst_size),
                        mode='bicubic',
                        align_corners=False).reshape(
                            dst_heads, dst_length).permute(1, 0).contiguous()

            if value.shape != target[key].shape:
                skipped.append(key)
                continue
            loadable[key] = value

        incompatible = self.load_state_dict(loadable, strict=False)
        print_log(
            f'Loaded {len(loadable)} GFM encoder tensors from '
            f'{checkpoint_path}; skipped {len(skipped)} incompatible or '
            'non-backbone tensors.')
        print_log(str(incompatible))

        del checkpoint, source, loadable
        gc.collect()

