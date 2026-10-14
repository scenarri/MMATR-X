"""Loading and augmentation transforms for paired RGB-SAR images."""

from typing import Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.datasets.transforms.transforms import PhotoMetricDistortion
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPairedImageFromFile(BaseTransform):
    """Load registered RGB and SAR images and concatenate their channels.

    Concatenating before geometric augmentation lets the standard MMSeg
    transforms resize, crop and flip both modalities with exactly the same
    sampled parameters.
    """

    def __init__(self,
                 to_float32: bool = False,
                 rgb_color_type: str = 'color',
                 sar_color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.rgb_color_type = rgb_color_type
        self.sar_color_type = sar_color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty
        self.backend_args = backend_args.copy() if backend_args else None

    def _load(self, path: str, color_type: str) -> np.ndarray:
        img_bytes = fileio.get(path, backend_args=self.backend_args)
        image = mmcv.imfrombytes(
            img_bytes, flag=color_type, backend=self.imdecode_backend)
        if image is None:
            raise ValueError(f'Failed to decode image: {path}')
        if image.ndim == 2:
            image = image[..., None]
        return image

    def transform(self, results: dict) -> Optional[dict]:
        rgb_path = results['img_path']
        sar_path = results['img_path2']
        try:
            rgb = self._load(rgb_path, self.rgb_color_type)
            sar = self._load(sar_path, self.sar_color_type)
        except Exception:
            if self.ignore_empty:
                return None
            raise

        if rgb.shape[:2] != sar.shape[:2]:
            raise ValueError(
                'Paired RGB/SAR images must have identical spatial shapes, '
                f'but got RGB {rgb.shape} and SAR {sar.shape}: '
                f'{rgb_path}, {sar_path}')

        if self.to_float32:
            rgb = rgb.astype(np.float32)
            sar = sar.astype(np.float32)

        results['img'] = np.ascontiguousarray(
            np.concatenate((rgb, sar), axis=-1))
        results['img_shape'] = rgb.shape[:2]
        results['ori_shape'] = rgb.shape[:2]
        results['modality_channels'] = (rgb.shape[-1], sar.shape[-1])
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f"rgb_color_type='{self.rgb_color_type}', "
                f"sar_color_type='{self.sar_color_type}', "
                f"imdecode_backend='{self.imdecode_backend}', "
                f'ignore_empty={self.ignore_empty}, '
                f'backend_args={self.backend_args})')


@TRANSFORMS.register_module()
class PairedPhotoMetricDistortion(PhotoMetricDistortion):
    """Apply MMSeg photometric distortion to selected modalities only.

    Geometry has already been shared by concatenating the images. RGB and SAR
    are distorted independently, so both modalities use the same augmentation
    procedure while sampling their own photometric parameters.
    """

    def __init__(self,
                 rgb_channels: int = 3,
                 sar_channels: int = 3,
                 apply_to_rgb: bool = True,
                 apply_to_sar: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.rgb_channels = rgb_channels
        self.sar_channels = sar_channels
        self.apply_to_rgb = apply_to_rgb
        self.apply_to_sar = apply_to_sar
        if apply_to_rgb and rgb_channels != 3:
            raise ValueError('RGB photometric distortion requires 3 channels')
        if apply_to_sar and sar_channels != 3:
            raise ValueError('SAR photometric distortion requires 3 channels')

    def _distort(self, image: np.ndarray) -> np.ndarray:
        return super().transform(dict(img=image))['img']

    def transform(self, results: dict) -> dict:
        image = results['img']
        expected_channels = self.rgb_channels + self.sar_channels
        if image.shape[-1] != expected_channels:
            raise ValueError(
                f'Expected {expected_channels} concatenated channels, but '
                f'got {image.shape[-1]}')

        rgb = image[..., :self.rgb_channels]
        sar = image[..., self.rgb_channels:]
        if self.apply_to_rgb:
            rgb = self._distort(rgb)
        if self.apply_to_sar:
            sar = self._distort(sar)
        results['img'] = np.ascontiguousarray(
            np.concatenate((rgb, sar), axis=-1))
        return results
