"""Data preprocessor with modality-specific normalization."""

from numbers import Number
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class DualModalSegDataPreProcessor(SegDataPreProcessor):
    """Normalize concatenated RGB/SAR input with separate statistics."""

    def __init__(
        self,
        rgb_mean: Sequence[Number],
        rgb_std: Sequence[Number],
        sar_mean: Sequence[Number],
        sar_std: Sequence[Number],
        modality_channels: Tuple[int, int] = (3, 3),
        bgr_to_rgb: Tuple[bool, bool] = (True, True),
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: Optional[dict] = None,
    ) -> None:
        rgb_channels, sar_channels = modality_channels
        if len(rgb_mean) != rgb_channels or len(rgb_std) != rgb_channels:
            raise ValueError('RGB statistics must match rgb_channels')
        if len(sar_mean) != sar_channels or len(sar_std) != sar_channels:
            raise ValueError('SAR statistics must match sar_channels')
        if len(bgr_to_rgb) != 2:
            raise ValueError('bgr_to_rgb must contain one flag per modality')
        if any(bgr_to_rgb) and (rgb_channels, sar_channels) != (3, 3):
            raise ValueError('BGR-to-RGB conversion is only defined for 3+3 channels')

        super().__init__(
            mean=list(rgb_mean) + list(sar_mean),
            std=list(rgb_std) + list(sar_std),
            size=size,
            size_divisor=size_divisor,
            pad_val=pad_val,
            seg_pad_val=seg_pad_val,
            bgr_to_rgb=False,
            batch_augments=batch_augments,
            test_cfg=test_cfg)
        self.modality_channels = tuple(modality_channels)
        self.modality_bgr_to_rgb = tuple(bgr_to_rgb)

    def _convert_channels(self, image: torch.Tensor) -> torch.Tensor:
        expected = sum(self.modality_channels)
        if image.size(0) != expected:
            raise ValueError(
                f'Expected {expected} channels, but got {image.size(0)}')
        rgb_channels = self.modality_channels[0]
        rgb = image[:rgb_channels]
        sar = image[rgb_channels:]
        if self.modality_bgr_to_rgb[0]:
            rgb = rgb[[2, 1, 0], ...]
        if self.modality_bgr_to_rgb[1]:
            sar = sar[[2, 1, 0], ...]
        return torch.cat((rgb, sar), dim=0)

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        data = self.cast_data(data)
        inputs = [self._convert_channels(item).float()
                  for item in data['inputs']]
        data_samples = data.get('data_samples', None)

        if self._enable_normalize:
            inputs = [(item - self.mean) / self.std for item in inputs]

        if training:
            if data_samples is None:
                raise ValueError('data_samples are required during training')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)
            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            image_size = inputs[0].shape[1:]
            if not all(item.shape[1:] == image_size for item in inputs):
                raise ValueError('All images in an inference batch must match')
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                if data_samples is not None:
                    for sample, pad_info in zip(data_samples, padded_samples):
                        sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
