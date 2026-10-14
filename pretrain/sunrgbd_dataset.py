import os
import os.path as osp
import albumentations
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import copy
from torch.utils.data import Dataset, Sampler, DataLoader
import torch.distributed as dist
from typing import Optional, List, Iterator
import time

stats = {
    "sunrgbd_pretrain": {
        "depth": ((0.3022, 0.3022, 0.3022), (0.1536, 0.1536, 0.1536)),
        "rgb": ((0.4964, 0.4668, 0.4409), (0.2777, 0.2861, 0.2909))},
}


class SUNRGBD_Dataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            mode: str = "pair",  # "sar", "rgb", or "pair"
            img_size: int = 224,
            use_aug: bool = True,
    ):

        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.use_aug = use_aug
        self.stats = stats
        self.epoch = 0

        self.samples = []
        self._collect_samples()

        self.paired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 1]
        self.unpaired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 0]

        # Setup augmentations
        self.pair_transform, self.single_transform = self._build_transforms()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):

            if dataset_name == '.ipynb_checkpoints':
                continue

            dataset_path = osp.join(self.root_dir, dataset_name)
            num = 0

            split_file = osp.join(dataset_path, "train.txt")
            with open(split_file, 'r') as f:
                for line in f:
                    img_name = line.strip()
                    if not img_name.endswith('.png'):
                        img_name += '.png'

                    sample = {}
                    sample['rgb'] = osp.join(dataset_path, "RGB", img_name)
                    sample['depth'] = osp.join(dataset_path, "Depth", img_name)
                    sample['dataset'] = dataset_name
                    sample['paired'] = 1
                    self.samples.append(sample)
                    num += 1

            print(f"{num} samples found in {dataset_path}")

    def _build_transforms(self):
        """Build albumentations transform pipeline"""
        interpolation = cv2.INTER_CUBIC
        if self.use_aug:
            pair_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation
                ),
                A.HorizontalFlip(p=0.5),
            ], additional_targets={'rgb': 'image'})

            single_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation
                ),
                A.HorizontalFlip(p=0.5)])
        else:
            pair_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ], additional_targets={'rgb': 'image'})

            single_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

        return pair_transform, single_transform

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).float()

    def _normalize(self, tensor: torch.Tensor, dataset: str, modality: str) -> torch.Tensor:
        mean, std = self.stats[dataset][modality]
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        return (tensor - mean_tensor) / std_tensor, mean_tensor, std_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):

        rgb = self.samples[idx]['rgb']
        sar = self.samples[idx]['depth']
        paired = self.samples[idx]['paired']
        dataset_name = self.samples[idx]['dataset']

        # Read images
        rgb_img = Image.open(rgb).convert('RGB')
        depth_img = Image.open(sar).convert('RGB')

        # Apply transforms
        if paired == 1:
            transformed = self.pair_transform(image=np.array(depth_img), rgb=np.array(rgb_img))
            rgb_trans = transformed["rgb"]
            depth_trans = transformed["image"]
        else:
            rgb_trans = self.single_transform(image=np.array(rgb_img))["image"]
            depth_trans = self.single_transform(image=np.array(depth_img))["image"]

        rgb_tensor = self._to_tensor(rgb_trans)
        depth_tensor = self._to_tensor(depth_trans)

        # Apply dataset-specific normalization
        rgb_tensor_norm, opt_mean, opt_std = self._normalize(rgb_tensor, dataset_name, "rgb")
        depth_tensor_norm, sar_mean, sar_std = self._normalize(depth_tensor, dataset_name, "depth")

        return {'normalized': torch.cat([rgb_tensor_norm, depth_tensor_norm], dim=0),
                'original': torch.cat([rgb_tensor, depth_tensor], dim=0),
                'dataset': dataset_name,
                'paired': paired,
                'stats': [opt_mean, opt_std, sar_mean, sar_std]}


class DeCURBrightnessContrast(A.ImageOnlyTransform):
    """
    Exact Albumentations implementation of DeCUR:

    RandomApply([
        RandomBrightness(0.4),
        RandomContrast(0.4)
    ], p=0.8)
    """

    def __init__(
            self,
            brightness=0.4,
            contrast=0.4,
            always_apply=False,
            p=0.8,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness = brightness
        self.contrast = contrast

    def get_params(self):
        brightness_factor = np.random.uniform(
            max(0, 1 - self.brightness),
            1 + self.brightness,
        )

        contrast_factor = np.random.uniform(
            max(0, 1 - self.contrast),
            1 + self.contrast,
        )

        return {
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
        }

    def apply(
            self,
            img,
            brightness_factor=1.0,
            contrast_factor=1.0,
            **params,
    ):
        # -------------------------
        # DeCUR RandomBrightness
        # -------------------------
        img = img * brightness_factor
        img = img.astype(np.uint8)

        # -------------------------
        # DeCUR RandomContrast
        # -------------------------
        mean = np.mean(img, axis=(0, 1))

        img = (
                (img - mean) * contrast_factor
                + mean
        ).astype(np.uint8)

        return img

    def get_transform_init_args_names(self):
        return ("brightness", "contrast")


class SUNRGBD_Dataset_DECUR(Dataset):
    def __init__(
            self,
            root_dir: str,
            mode: str = "pair",  # "sar", "rgb", or "pair"
            img_size: int = 224,
            use_aug: bool = True,
    ):

        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.use_aug = use_aug
        self.stats = stats
        self.epoch = 0

        self.samples = []
        self._collect_samples()

        self.paired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 1]
        self.unpaired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 0]

        # Setup augmentations
        self.rgb_transform, self.depth_transform = self._build_transforms()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):

            if dataset_name == '.ipynb_checkpoints':
                continue

            dataset_path = osp.join(self.root_dir, dataset_name)
            num = 0

            split_file = osp.join(dataset_path, "train.txt")
            with open(split_file, 'r') as f:
                for line in f:
                    img_name = line.strip()
                    if not img_name.endswith('.png'):
                        img_name += '.png'

                    sample = {}
                    sample['rgb'] = osp.join(dataset_path, "RGB", img_name)
                    sample['depth'] = osp.join(dataset_path, "Depth", img_name)
                    sample['dataset'] = dataset_name
                    sample['paired'] = 1
                    self.samples.append(sample)
                    num += 1

            print(f"{num} samples found in {dataset_path}")

    def _build_transforms(self):
        """Build DeCUR-style augmentation with Albumentations 1.4.18."""
        interpolation = cv2.INTER_CUBIC
        if self.use_aug:
            # Optical / RGB
            rgb_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation, ),
                DeCURBrightnessContrast(
                    brightness=0.4,
                    contrast=0.4,
                    p=0.8, ),
                A.ToGray(
                    num_output_channels=3,
                    p=0.2, ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=(0.1, 2.0),
                    p=0.5, ),
                A.Solarize(
                    threshold=128,
                    p=0.2, ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5), ])

            # depth
            depth_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation, ),
                A.ToGray(
                    num_output_channels=3,
                    p=0.2, ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            rgb_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

            depth_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

        return rgb_transform, depth_transform

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).float()

    def _normalize(self, tensor: torch.Tensor, dataset: str, modality: str) -> torch.Tensor:
        """Apply dataset-specific normalization"""
        # if dataset in self.stats and modality in self.stats[dataset]:
        mean, std = self.stats[dataset][modality]
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        return (tensor - mean_tensor) / std_tensor, mean_tensor, std_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):

        rgb = self.samples[idx]['rgb']
        sar = self.samples[idx]['depth']
        paired = self.samples[idx]['paired']
        dataset_name = self.samples[idx]['dataset']

        # Read images
        rgb_img = Image.open(rgb).convert('RGB')
        depth_img = Image.open(sar).convert('RGB')

        # Apply transforms
        if paired == 1:
            rgb_np = np.array(rgb_img)
            depth_np = np.array(depth_img)

            rgb_trans1 = self.rgb_transform(image=rgb_np)["image"]
            rgb_trans2 = self.rgb_transform(image=rgb_np)["image"]

            depth_trans1 = self.depth_transform(image=depth_np)["image"]
            depth_trans2 = self.depth_transform(image=depth_np)["image"]
        else:
            raise NotImplementedError

        rgb_tensor_1 = self._to_tensor(rgb_trans1)
        rgb_tensor_2 = self._to_tensor(rgb_trans2)
        depth_tensor_1 = self._to_tensor(depth_trans1)
        depth_tensor_2 = self._to_tensor(depth_trans2)

        # Apply dataset-specific normalization
        rgb_tensor_norm_1, _, _ = self._normalize(rgb_tensor_1, dataset_name, "rgb")
        rgb_tensor_norm_2, _, _ = self._normalize(rgb_tensor_2, dataset_name, "rgb")
        depth_tensor_norm_1, _, _ = self._normalize(depth_tensor_1, dataset_name, "depth")
        depth_tensor_norm_2, _, _ = self._normalize(depth_tensor_2, dataset_name, "depth")

        return {
            'normalized': torch.cat([rgb_tensor_norm_1, rgb_tensor_norm_2, depth_tensor_norm_1, depth_tensor_norm_2],
                                    dim=0),
            'original': torch.cat([rgb_tensor_1, rgb_tensor_2, depth_tensor_1, depth_tensor_2], dim=0),
            'dataset': dataset_name,
            'paired': paired, }


class Single_Modality_Dataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 modality: str = "rgb",
                 img_size: int = 224,
                 use_aug: bool = True):

        self.root_dir = root_dir
        self.modality = modality
        self.img_size = img_size
        self.use_aug = use_aug
        self.stats = stats
        self.base_transform = self._build_transforms()

        self.samples = []
        self._collect_samples()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):
            dataset_path = osp.join(self.root_dir, dataset_name)

            num = 0
            split_file = osp.join(dataset_path, "train.txt")
            with open(split_file, 'r') as f:
                for line in f:
                    num += 1
                    img_name = line.strip()
                    if not img_name.endswith('.png'):
                        img_name += '.png'

                    path = osp.join(dataset_path, 'RGB', img_name)
                    self.samples.append({
                        "path": path,
                        "modality": self.modality,
                        "dataset": dataset_name})

            print(f'Found {num} images in {dataset_name}')

    def _build_transforms(self):
        """Build albumentations transform pipeline"""
        interpolation = cv2.INTER_CUBIC
        if self.use_aug:
            base_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation
                ),
                A.HorizontalFlip(p=0.5)])
        else:
            base_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

        return base_transform

    def _to_tensor(self, img):
        return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).float()

    def _normalize(self, tensor, dataset, modality):
        """Apply dataset-specific normalization"""
        if dataset in self.stats and modality in self.stats[dataset]:
            mean, std = self.stats[dataset][modality]
            mean_tensor = torch.tensor(mean).view(-1, 1, 1)
            std_tensor = torch.tensor(std).view(-1, 1, 1)
            return (tensor - mean_tensor) / std_tensor
        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dataset_name = sample["dataset"]
        modality = sample["modality"]

        # Read images
        img = Image.open(sample["path"]).convert('RGB')

        # Apply transforms
        img_trans = self.base_transform(image=np.array(img))["image"]
        img_tensor = self._to_tensor(np.array(img_trans))

        # Apply dataset-specific normalization
        img_tensor_norm = self._normalize(img_tensor, dataset_name, modality)

        return img_tensor_norm