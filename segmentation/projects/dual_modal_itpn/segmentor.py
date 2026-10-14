"""Dual-modal encoder-decoder segmentor."""

from typing import List

from torch import Tensor

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class DualModalEncoderDecoder(EncoderDecoder):
    """Split a channel-concatenated input before dual-modal feature encoding."""

    def __init__(self, rgb_channels: int = 3, sar_channels: int = 3,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.rgb_channels = rgb_channels
        self.sar_channels = sar_channels

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        expected_channels = self.rgb_channels + self.sar_channels
        if inputs.ndim != 4 or inputs.size(1) != expected_channels:
            raise ValueError(
                'DualModalEncoderDecoder expects inputs shaped '
                f'[B, {expected_channels}, H, W], but got '
                f'{tuple(inputs.shape)}')
        rgb = inputs[:, :self.rgb_channels]
        sar = inputs[:, self.rgb_channels:]
        features = self.backbone(rgb, sar)
        if self.with_neck:
            features = self.neck(features)
        return features
