"""Numerically compare dual-modal and original single-modal iTPN paths.

This is a standalone integration check rather than a unit test with mocked
weights. It loads the supplied checkpoints, builds the official experiment
configs and compares all four output pyramid levels.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import MMLogger
from mmengine.registry import DefaultScope
from torch import Tensor

from mmseg.models.backbones.iTPN_pixel_dualembed import iTPN_pixel_dualembed
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS

import projects.dual_modal_itpn  # noqa: F401


SHARED_CONFIG = ROOT / 'configs/_major_/ddhr_rgb_sar_ownstats_shared.py'
INDEPENDENT_CONFIG = (
    ROOT / 'configs/_major_/ddhr_rgb_sar_ownstats_independent.py')

ITPN_COMMON = dict(
    img_size=256,
    patch_size=16,
    embed_dim=512,
    mlp_depth1=3,
    mlp_depth2=3,
    depth=24,
    num_heads=8,
    mlp_ratio=4,
    fpn_dim=256,
    fpn_depth=1,
    qkv_bias=True,
    ape=True,
    rpe=False,
    drop_path_rate=0.1,
    num_outs=4,
    use_checkpoint=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-checkpoint', required=True)
    parser.add_argument('--rgb-checkpoint', required=True)
    parser.add_argument('--sar-checkpoint', required=True)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=20260914)
    # Concatenating modalities changes the CUDA GEMM batch shape. Although the
    # computation is mathematically identical, reduction order can introduce
    # float32 differences around 1e-4 on recent GPUs.
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-5)
    return parser.parse_args()


def clean_up() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_original(modality: str, checkpoint_path: str,
                   device: torch.device) -> iTPN_pixel_dualembed:
    init_cfg = ConfigDict(type='Pretrained', checkpoint=checkpoint_path)
    model = iTPN_pixel_dualembed(
        modality=modality, init_cfg=init_cfg, **ITPN_COMMON)
    model.init_weights()
    return model.eval().to(device)


@torch.inference_mode()
def original_features(modality: str, checkpoint_path: str, image: Tensor,
                      device: torch.device) -> Sequence[Tensor]:
    model = build_original(modality, checkpoint_path, device)
    outputs = tuple(item.cpu() for item in model(image.to(device)))
    del model
    clean_up()
    return outputs


def build_shared_segmentor(checkpoint_path: str, initialize: bool = True):
    cfg = Config.fromfile(str(SHARED_CONFIG))
    cfg.model.backbone.init_cfg.checkpoint = checkpoint_path
    cfg.model.backbone.use_checkpoint = False
    model = MODELS.build(cfg.model)
    if initialize:
        model.init_weights()
    return model


def build_independent_segmentor(rgb_checkpoint: str, sar_checkpoint: str):
    cfg = Config.fromfile(str(INDEPENDENT_CONFIG))
    cfg.model.backbone.rgb_backbone.init_cfg.checkpoint = rgb_checkpoint
    cfg.model.backbone.sar_backbone.init_cfg.checkpoint = sar_checkpoint
    cfg.model.backbone.rgb_backbone.use_checkpoint = False
    cfg.model.backbone.sar_backbone.use_checkpoint = False
    model = MODELS.build(cfg.model)
    model.init_weights()
    return model


def make_inputs(seed: int) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    rgb_bgr = torch.randint(
        0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator)
    sar_bgr = torch.randint(
        0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator)
    paired_bgr = torch.cat((rgb_bgr, sar_bgr), dim=0)
    return rgb_bgr, sar_bgr, paired_bgr


def verify_preprocessing(model, rgb_bgr: Tensor, sar_bgr: Tensor,
                         paired_bgr: Tensor) -> Tensor:
    paired = model.data_preprocessor(
        dict(inputs=[paired_bgr]), training=False)['inputs']
    rgb_preprocessor = SegDataPreProcessor(
        mean=[162.457, 164.329, 166.7645],
        std=[41.7551, 39.3668, 37.239],
        bgr_to_rgb=True)
    sar_preprocessor = SegDataPreProcessor(
        mean=[50.774, 50.774, 50.774],
        std=[49.6096, 49.6096, 49.6096],
        bgr_to_rgb=True)
    rgb = rgb_preprocessor(
        dict(inputs=[rgb_bgr]), training=False)['inputs']
    sar = sar_preprocessor(
        dict(inputs=[sar_bgr]), training=False)['inputs']

    assert torch.equal(paired[:, :3], rgb), (
        'Dual-modal RGB preprocessing differs from the original pipeline')
    assert torch.equal(paired[:, 3:], sar), (
        'Dual-modal SAR preprocessing differs from the original pipeline')
    print('preprocessor: exact match')
    return paired


@torch.inference_mode()
def dual_features(build_model: Callable, paired: Tensor,
                  device: torch.device) -> Sequence[Tensor]:
    model = build_model().eval().to(device)
    outputs = tuple(item.cpu() for item in model.extract_feat(
        paired.to(device)))
    del model
    clean_up()
    return outputs


def compare_pyramids(name: str, actual: Sequence[Tensor],
                     rgb_reference: Sequence[Tensor],
                     sar_reference: Sequence[Tensor], atol: float,
                     rtol: float) -> None:
    assert len(actual) == len(rgb_reference) == len(sar_reference) == 4
    passed = True
    for level, (output, rgb, sar) in enumerate(
            zip(actual, rgb_reference, sar_reference)):
        expected = torch.cat((rgb, sar), dim=1)
        difference = (output - expected).abs()
        level_passed = torch.allclose(
            output, expected, atol=atol, rtol=rtol)
        passed &= level_passed
        print(
            f'{name} level {level}: shape={tuple(output.shape)}, '
            f'max_abs={difference.max().item():.8g}, '
            f'mean_abs={difference.mean().item():.8g}, '
            f'allclose={level_passed}')
    if not passed:
        raise AssertionError(f'{name} pyramid is not numerically equivalent')


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    DefaultScope.get_instance('dual_modal_equivalence', scope_name='mmseg')
    MMLogger.get_instance('dual_modal_equivalence', log_level='WARNING')
    torch.backends.cuda.matmul.allow_tf32 = False

    rgb_bgr, sar_bgr, paired_bgr = make_inputs(args.seed)

    # Build once to verify the new modality-specific preprocessing. Both
    # experiment configs use exactly the same preprocessor configuration.
    preprocessor_model = build_shared_segmentor(
        args.shared_checkpoint, initialize=False)
    paired = verify_preprocessing(
        preprocessor_model, rgb_bgr, sar_bgr, paired_bgr)
    del preprocessor_model
    clean_up()

    shared_actual = dual_features(
        lambda: build_shared_segmentor(args.shared_checkpoint), paired,
        device)
    shared_rgb = original_features(
        'RGB', args.shared_checkpoint, paired[:, :3], device)
    shared_sar = original_features(
        'SAR', args.shared_checkpoint, paired[:, 3:], device)
    compare_pyramids('shared', shared_actual, shared_rgb, shared_sar,
                     args.atol, args.rtol)
    del shared_actual, shared_rgb, shared_sar
    clean_up()

    independent_actual = dual_features(
        lambda: build_independent_segmentor(
            args.rgb_checkpoint, args.sar_checkpoint), paired, device)
    independent_rgb = original_features(
        'RGB', args.rgb_checkpoint, paired[:, :3], device)
    independent_sar = original_features(
        'SAR', args.sar_checkpoint, paired[:, 3:], device)
    compare_pyramids('independent', independent_actual, independent_rgb,
                     independent_sar, args.atol, args.rtol)

    print('all numerical-equivalence checks passed')


if __name__ == '__main__':
    main()
