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
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from typing import Optional, List, Iterator
import time

stats = {
    "SpaceNet6": {
        "sar": ((0.3552, 0.3552, 0.3552), (0.2093, 0.2093, 0.2093)),
        "rgb": ((0.3712, 0.3928, 0.3576), (0.1871, 0.1627, 0.1723))},
    "DFC23_Track2": {
        "sar": ((0.1926, 0.1926, 0.1926), (0.1986, 0.1986, 0.1986)),
        "rgb": ((0.3159, 0.3418, 0.2800), (0.1797, 0.1655, 0.1688))},
    "DFC23_Track1": {
        "sar": ((0.1814, 0.1814, 0.1814), (0.1975, 0.1975, 0.1975)),
        "rgb": ((0.3240, 0.3476, 0.2904), (0.2159, 0.1874, 0.1739))},
    "DFC25": {
        "sar": ((0.1941, 0.1941, 0.1941), (0.1369, 0.1369, 0.1369)),
        "rgb": ((0.4920, 0.5066, 0.4548), (0.2164, 0.1811, 0.1765))},
    "QXSLAB_SARPORT": {
        "sar": ((0.2707, 0.2707, 0.2707), (0.2359, 0.2359, 0.2359)),
        "rgb": ((0.4371, 0.4139, 0.4133), (0.2016, 0.1717, 0.1654))},
    "OGSOD-2": {
        "sar": ((0.4770, 0.4770, 0.4770), (0.2693, 0.2693, 0.2693)),
        "rgb": ((0.3544, 0.3587, 0.3384), (0.2237, 0.2099, 0.2112))},
    "PIE-RGB-SAR": {
        "sar": ((0.3617, 0.3617, 0.3617), (0.2141, 0.2141, 0.2141)),
        "rgb": ((0.3545, 0.3812, 0.3122), (0.2184, 0.1870, 0.1907))},
    "YESeg-OPT-SAR": {
        "sar": ((0.1598, 0.1598, 0.1598), (0.1344, 0.1344, 0.1344)),
        "rgb": ((0.3878, 0.3814, 0.3735), (0.1493, 0.1443, 0.1418))},
    "DDHR": {
        "sar": ((0.2365, 0.2365, 0.2365), (0.2034, 0.2034, 0.2034)),
        "rgb": ((0.2243, 0.2357, 0.2443), (0.1463, 0.1329, 0.1274))},
    "WHU-OPT-SAR": {
        "sar": ((0.2119, 0.2119, 0.2119), (0.1911, 0.1911, 0.1911)),
        "rgb": ((0.1639, 0.1527, 0.1260), (0.0723, 0.0740, 0.0703))},
    "SS4EO-S12": {
        "sar": ((0.3819, 0.3819, 0.3819), (0.2224, 0.2224, 0.2224)),
        "rgb": ((0.4842, 0.4813, 0.4785), (0.1963, 0.1732, 0.1483))},
    "GFGE_SOdataset": {
        "sar": ((0.3929, 0.3929, 0.3929), (0.2324, 0.2324, 0.2324)),
        "rgb": ((0.4005, 0.3968, 0.3373), (0.2430, 0.1730, 0.1549))},
    "EarthMiss": {
        "sar": ((0.2871, 0.2871, 0.2871), (0.2626, 0.2626, 0.2626)),
        "rgb": ((0.4147, 0.4097, 0.3867), (0.2342, 0.2022, 0.2002))},
    "osdataset": {
        "sar": ((0.2683, 0.2683, 0.2683), (0.1898, 0.1898, 0.1898)),
        "rgb": ((0.4071, 0.4227, 0.3414), (0.237, 0.2035, 0.2038))},
    "Unpaired": {
        "sar": ((0.1738, 0.1738, 0.1738), (0.2226, 0.2226, 0.2226)),
        "rgb": ((0.3865, 0.4117, 0.3632), (0.2093, 0.1803, 0.1748))}
}


class PAIRED_Dataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            mode: str = "pair",  # "sar", "rgb", or "pair"
            exclude_test: Optional[List[str]] = None,
            exclude_dataset: Optional[List[str]] = None,
            img_size: int = 224,
            use_aug: bool = True,
            resume=None
    ):
        """
        PyTorch Dataset for multi-modal satellite imagery with SAR and RGB pairs.

        Args:
            root_dir: Root directory containing dataset subfolders
            mode: Data modality to return ("sar", "rgb", "pair")
            exclude_test: List of dataset names to exclude test sets from
            stats: Dictionary with dataset-specific normalization stats
            img_size: Target image size (square)
            use_aug: Apply data augmentation if True
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.use_aug = use_aug
        self.exclude_test = exclude_test or []
        self.exclude_dataset = exclude_dataset or []
        self.stats = stats
        self.epoch = 0

        # Collect all valid samples
        if resume is None:
            self.samples = []
            self._collect_samples()

        self.paired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 1]
        self.unpaired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 0]

        # Setup augmentations
        self.pair_transform, self.single_transform = self._build_transforms()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):
            # Check if we should exclude this dataset
            if dataset_name in self.exclude_dataset:
                continue

            dataset_path = osp.join(self.root_dir, dataset_name)
            num = 0
            if dataset_name != 'Unpaired':
                # Check if we should exclude this dataset's test set
                exclude_this_test = dataset_name in self.exclude_test
                # Process both train and test splits
                for split in ["train", "val", "test"]:
                    if exclude_this_test and split == "test":
                        continue  # Skip test set for excluded datasets
                    split_file = osp.join(dataset_path, f"{split}.txt")
                    if not os.path.exists(split_file):
                        continue

                    # Read image names for this split
                    with open(split_file, 'r') as f:
                        for line in f:
                            img_name = line.strip()
                            if not img_name.endswith('.png'):
                                img_name += '.png'

                            sample = {}
                            sample['rgb'] = osp.join(dataset_path, "RGB", img_name)
                            sample['sar'] = osp.join(dataset_path, "SAR", img_name)
                            sample['dataset'] = dataset_name
                            sample['paired'] = 1
                            self.samples.append(sample)
                            num += 1
            else:
                sar_imgs = os.listdir(osp.join(dataset_path, 'SAR'))
                rgb_imgs = os.listdir(osp.join(dataset_path, 'RGB'))
                assert len(sar_imgs) == len(rgb_imgs)
                for sar, rgb in zip(sar_imgs, rgb_imgs):
                    sample = {}
                    sample['rgb'] = osp.join(dataset_path, "RGB", rgb)
                    sample['sar'] = osp.join(dataset_path, "SAR", sar)
                    sample['dataset'] = dataset_name
                    sample['paired'] = 0
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

    def shuffle(self, ):
        time_start = time.time()
        if len(self.unpaired_indices) == 0:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print("No unpaired samples to shuffle.")
            return

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        if rank == 0:
            random.seed(self.epoch + 1027)
            unpaired_sar_paths = [self.samples[idx]['sar'] for idx in self.unpaired_indices]
            random.shuffle(unpaired_sar_paths)
        else:
            unpaired_sar_paths = [None] * len(self.unpaired_indices)

        # 广播打乱后的unpaired_sar_paths给所有进程
        if world_size > 1:
            dist.broadcast_object_list(unpaired_sar_paths, src=0)

        # 使用屏障确保所有进程在此处同步
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        for i, idx in enumerate(self.unpaired_indices):
            self.samples[idx]['sar'] = unpaired_sar_paths[i]

        time_end = time.time()
        if rank == 0:
            print(f"Shuffled {len(self.unpaired_indices)} unpaired samples with {time_end - time_start} seconds.")

    def __getitem__(self, idx):

        rgb = self.samples[idx]['rgb']
        sar = self.samples[idx]['sar']
        paired = self.samples[idx]['paired']
        dataset_name = self.samples[idx]['dataset']

        # Read images
        rgb_img = Image.open(rgb).convert('RGB')
        sar_img = Image.open(sar).convert('RGB')

        # handle abnormal img in ssl4eo
        if dataset_name == 'SSL4EO':
            rgb_shape = np.array(rgb_img).shape
            if rgb_shape[0] != rgb_shape[1]:
                max_hw = max(rgb_shape[0], rgb_shape[1])
                rgb_img = rgb_img.resize((max_hw, max_hw), 0)
                sar_img = sar_img.resize((max_hw, max_hw), 0)
            sar_shape = np.array(sar_img).shape
            if sar_shape[0] != sar_shape[1]:
                max_hw = max(sar_shape[0], sar_shape[1])
                rgb_img = rgb_img.resize((max_hw, max_hw), 0)
                sar_img = sar_img.resize((max_hw, max_hw), 0)
            rgb_shape = np.array(rgb_img).shape
            if rgb_shape[0] == 263 and rgb_shape[1] == 263:
                rgb_img = rgb_img.resize((264, 264), 0)
            sar_shape = np.array(sar_img).shape
            if sar_shape[0] == 263 and sar_shape[1] == 263:
                sar_img = sar_img.resize((264, 264), 0)

        # Apply transforms
        if paired == 1:
            transformed = self.pair_transform(image=np.array(sar_img), rgb=np.array(rgb_img))
            rgb_trans = transformed["rgb"]
            sar_trans = transformed["image"]
        else:
            rgb_trans = self.single_transform(image=np.array(rgb_img))["image"]
            sar_trans = self.single_transform(image=np.array(sar_img))["image"]

        rgb_tensor = self._to_tensor(rgb_trans)
        sar_tensor = self._to_tensor(sar_trans)

        # Apply dataset-specific normalization
        rgb_tensor_norm, opt_mean, opt_std = self._normalize(rgb_tensor, dataset_name, "rgb")
        sar_tensor_norm, sar_mean, sar_std = self._normalize(sar_tensor, dataset_name, "sar")

        return {'normalized': torch.cat([rgb_tensor_norm, sar_tensor_norm], dim=0),
                'original': torch.cat([rgb_tensor, sar_tensor], dim=0),
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


class PAIRED_Dataset_DECUR(Dataset):
    def __init__(
            self,
            root_dir: str,
            mode: str = "pair",  # "sar", "rgb", or "pair"
            exclude_test: Optional[List[str]] = None,
            exclude_dataset: Optional[List[str]] = None,
            img_size: int = 224,
            use_aug: bool = True,
            resume=None
    ):

        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.use_aug = use_aug
        self.exclude_test = exclude_test or []
        self.exclude_dataset = exclude_dataset or []
        self.stats = stats
        self.epoch = 0

        # Collect all valid samples
        if resume is None:
            self.samples = []
            self._collect_samples()

        self.paired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 1]
        self.unpaired_indices = [i for i, sample in enumerate(self.samples) if sample['paired'] == 0]

        # Setup augmentations
        self.rgb_transform, self.sar_transform = self._build_transforms()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):
            # Check if we should exclude this dataset
            if dataset_name in self.exclude_dataset:
                continue

            dataset_path = osp.join(self.root_dir, dataset_name)
            num = 0
            if dataset_name != 'Unpaired':
                # Check if we should exclude this dataset's test set
                exclude_this_test = dataset_name in self.exclude_test
                # Process both train and test splits
                for split in ["train", "val", "test"]:
                    if exclude_this_test and split == "test":
                        continue  # Skip test set for excluded datasets
                    split_file = osp.join(dataset_path, f"{split}.txt")
                    if not os.path.exists(split_file):
                        continue

                    # Read image names for this split
                    with open(split_file, 'r') as f:
                        for line in f:
                            img_name = line.strip()
                            if not img_name.endswith('.png'):
                                img_name += '.png'

                            sample = {}
                            sample['rgb'] = osp.join(dataset_path, "RGB", img_name)
                            sample['sar'] = osp.join(dataset_path, "SAR", img_name)
                            sample['dataset'] = dataset_name
                            sample['paired'] = 1
                            self.samples.append(sample)
                            num += 1
            else:
                sar_imgs = os.listdir(osp.join(dataset_path, 'SAR'))
                rgb_imgs = os.listdir(osp.join(dataset_path, 'RGB'))
                assert len(sar_imgs) == len(rgb_imgs)
                for sar, rgb in zip(sar_imgs, rgb_imgs):
                    sample = {}
                    sample['rgb'] = osp.join(dataset_path, "RGB", rgb)
                    sample['sar'] = osp.join(dataset_path, "SAR", sar)
                    sample['dataset'] = dataset_name
                    sample['paired'] = 0
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

            # SAR
            sar_transform = A.Compose([
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.2, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    interpolation=interpolation, ),
                A.ToGray(
                    num_output_channels=3,
                    p=0.2, ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=(0.1, 2.0),
                    p=0.5, ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ])
        else:
            rgb_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

            sar_transform = A.Compose([
                A.Resize(self.img_size, self.img_size, interpolation=interpolation),
            ])

        return rgb_transform, sar_transform

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

    def shuffle(self, ):
        time_start = time.time()
        if len(self.unpaired_indices) == 0:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print("No unpaired samples to shuffle.")
            return

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        if rank == 0:
            random.seed(self.epoch + 1027)
            unpaired_sar_paths = [self.samples[idx]['sar'] for idx in self.unpaired_indices]
            random.shuffle(unpaired_sar_paths)
        else:
            unpaired_sar_paths = [None] * len(self.unpaired_indices)

        # 广播打乱后的unpaired_sar_paths给所有进程
        if world_size > 1:
            dist.broadcast_object_list(unpaired_sar_paths, src=0)

        # 使用屏障确保所有进程在此处同步
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        for i, idx in enumerate(self.unpaired_indices):
            self.samples[idx]['sar'] = unpaired_sar_paths[i]

        time_end = time.time()
        if rank == 0:
            print(f"Shuffled {len(self.unpaired_indices)} unpaired samples with {time_end - time_start} seconds.")

    def __getitem__(self, idx):

        rgb = self.samples[idx]['rgb']
        sar = self.samples[idx]['sar']
        paired = self.samples[idx]['paired']
        dataset_name = self.samples[idx]['dataset']

        # Read images
        rgb_img = Image.open(rgb).convert('RGB')
        sar_img = Image.open(sar).convert('RGB')

        # handle abnormal img in ssl4eo
        if dataset_name == 'SSL4EO':
            rgb_shape = np.array(rgb_img).shape
            if rgb_shape[0] != rgb_shape[1]:
                max_hw = max(rgb_shape[0], rgb_shape[1])
                rgb_img = rgb_img.resize((max_hw, max_hw), 0)
                sar_img = sar_img.resize((max_hw, max_hw), 0)
            sar_shape = np.array(sar_img).shape
            if sar_shape[0] != sar_shape[1]:
                max_hw = max(sar_shape[0], sar_shape[1])
                rgb_img = rgb_img.resize((max_hw, max_hw), 0)
                sar_img = sar_img.resize((max_hw, max_hw), 0)
            rgb_shape = np.array(rgb_img).shape
            if rgb_shape[0] == 263 and rgb_shape[1] == 263:
                rgb_img = rgb_img.resize((264, 264), 0)
            sar_shape = np.array(sar_img).shape
            if sar_shape[0] == 263 and sar_shape[1] == 263:
                sar_img = sar_img.resize((264, 264), 0)

        # Apply transforms
        if paired == 1:
            rgb_np = np.array(rgb_img)
            sar_np = np.array(sar_img)

            rgb_trans1 = self.rgb_transform(image=rgb_np)["image"]
            rgb_trans2 = self.rgb_transform(image=rgb_np)["image"]

            sar_trans1 = self.sar_transform(image=sar_np)["image"]
            sar_trans2 = self.sar_transform(image=sar_np)["image"]
        else:
            raise NotImplementedError

        rgb_tensor_1 = self._to_tensor(rgb_trans1)
        rgb_tensor_2 = self._to_tensor(rgb_trans2)
        sar_tensor_1 = self._to_tensor(sar_trans1)
        sar_tensor_2 = self._to_tensor(sar_trans2)

        # Apply dataset-specific normalization
        rgb_tensor_norm_1, _, _ = self._normalize(rgb_tensor_1, dataset_name, "rgb")
        rgb_tensor_norm_2, _, _ = self._normalize(rgb_tensor_2, dataset_name, "rgb")
        sar_tensor_norm_1, _, _ = self._normalize(sar_tensor_1, dataset_name, "sar")
        sar_tensor_norm_2, _, _ = self._normalize(sar_tensor_2, dataset_name, "sar")

        return {'normalized': torch.cat([rgb_tensor_norm_1, rgb_tensor_norm_2, sar_tensor_norm_1, sar_tensor_norm_2], dim=0),
                'original': torch.cat([rgb_tensor_1, rgb_tensor_2, sar_tensor_1, sar_tensor_2], dim=0),
                'dataset': dataset_name,
                'paired': paired, }


class Single_Modality_Dataset(Dataset):
    def __init__(self, root_dir: str, modality: str = "rgb", exclude_test: Optional[List[str]] = None,
                 exclude_dataset: Optional[List[str]] = None, img_size: int = 224, use_aug: bool = True,
                 resume=None):

        self.root_dir = root_dir
        self.modality = modality
        self.img_size = img_size
        self.use_aug = use_aug
        self.exclude_test = exclude_test or []
        self.exclude_dataset = exclude_dataset or []
        self.stats = stats
        self.base_transform = self._build_transforms()

        self.samples = []
        if resume is None:
            self._collect_samples()

    def _collect_samples(self):
        """Walk through directory structure to find valid image pairs"""
        for dataset_name in os.listdir(self.root_dir):
            dataset_path = osp.join(self.root_dir, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # Check if we should exclude this dataset
            exclude_this_dataset = dataset_name in self.exclude_dataset

            if exclude_this_dataset:
                continue

            num = 0
            if dataset_name != 'Unpaired':
                # Check if we should exclude this dataset's test set
                exclude_this_test = dataset_name in self.exclude_test
                # Process both train and test splits
                for split in ["train", "val", "test"]:
                    if exclude_this_test and split == "test":
                        continue  # Skip test set for excluded datasets
                    split_file = osp.join(dataset_path, f"{split}.txt")
                    if not os.path.exists(split_file):
                        continue
                    # Read image names for this split
                    with open(split_file, 'r') as f:
                        for line in f:
                            num += 1
                            img_name = line.strip()
                            if not img_name.endswith('.png'):
                                img_name += '.png'
                            if self.modality == "sar" or self.modality == "rgb":
                                path = osp.join(dataset_path, 'SAR' if self.modality == 'sar' else 'RGB', img_name)
                                self.samples.append({
                                    "path": path,
                                    "modality": self.modality,
                                    "dataset": dataset_name})
                            elif self.modality == "both":
                                path = osp.join(dataset_path, 'SAR', img_name)
                                self.samples.append({
                                    "path": path,
                                    "modality": 'sar',
                                    "dataset": dataset_name})
                                path = osp.join(dataset_path, 'RGB', img_name)
                                self.samples.append({
                                    "path": path,
                                    "modality": 'rgb',
                                    "dataset": dataset_name})
            else:
                if self.modality == 'sar':
                    for img in os.listdir(osp.join(dataset_path, 'SAR')):
                        self.samples.append({
                            "path": osp.join(dataset_path, 'SAR', img),
                            "modality": self.modality,
                            "dataset": dataset_name})
                        num += 1
                elif self.modality == 'rgb':
                    for img in os.listdir(osp.join(dataset_path, 'RGB')):
                        self.samples.append({
                            "path": osp.join(dataset_path, 'RGB', img),
                            "modality": self.modality,
                            "dataset": dataset_name})
                        num += 1
                elif self.modality == 'both':
                    for img in os.listdir(osp.join(dataset_path, 'SAR')):
                        self.samples.append({
                            "path": osp.join(dataset_path, 'SAR', img),
                            "modality": self.modality,
                            "dataset": dataset_name})
                        num += 1
                    for img in os.listdir(osp.join(dataset_path, 'RGB')):
                        self.samples.append({
                            "path": osp.join(dataset_path, 'RGB', img),
                            "modality": self.modality,
                            "dataset": dataset_name})
                        num += 1
                else:
                    raise ValueError("Unknown modality {}".format(self.modality))

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


class MixedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.indices = list(range(len(dataset)))

        self._generate_batches()

    def _generate_batches(self):
        paired_indices = copy.copy(self.dataset.paired_indices)
        unpaired_indices = copy.copy(self.dataset.unpaired_indices)

        if self.shuffle:
            if len(paired_indices) > 0:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                paired_indices = torch.tensor(paired_indices)[torch.randperm(len(paired_indices), generator=g)].tolist()

            if len(unpaired_indices) > 0:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + 11111)
                unpaired_indices = torch.tensor(unpaired_indices)[
                    torch.randperm(len(unpaired_indices), generator=g)].tolist()

        total_batches = []
        if len(paired_indices) > 0:
            paired_batches = []
            paired_batch_count = len(paired_indices) // self.batch_size
            for i in range(paired_batch_count):
                batch = paired_indices[i * self.batch_size:(i + 1) * self.batch_size]
                paired_batches.append(batch)
                total_batches.append(batch)

            if not self.drop_last:
                num_needed = len(paired_batches) % self.num_replicas
                if num_needed != 0:
                    for i in range(self.num_replicas - num_needed):
                        total_batches.append(paired_batches[i])

        if len(unpaired_indices) > 0:
            unpaired_batches = []
            unpaired_batch_count = len(unpaired_indices) // self.batch_size
            for i in range(unpaired_batch_count):
                batch = unpaired_indices[i * self.batch_size:(i + 1) * self.batch_size]
                unpaired_batches.append(batch)
                total_batches.append(batch)

            if not self.drop_last:
                num_needed = len(unpaired_batches) % self.num_replicas
                if num_needed != 0:
                    for i in range(self.num_replicas - num_needed):
                        total_batches.append(unpaired_batches[i])

        shuffled_total_batches = []
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + 729729)
            shufle_indices = torch.randperm(len(total_batches), generator=g).tolist()
        else:
            shufle_indices = torch.arange(len(total_batches)).tolist()
        for idx in shufle_indices:
            shuffled_total_batches.append(total_batches[idx])

        self.batches = []
        for i, batch in enumerate(shuffled_total_batches):
            if i % self.num_replicas == self.rank:
                self.batches.append(batch)

        if self.rank == 0:
            print(f"New distributed batch generated.")
        # 为当前replica分配批次

    def __iter__(self) -> Iterator[List[int]]:
        self._generate_batches()
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



