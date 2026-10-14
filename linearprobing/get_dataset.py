import os
import albumentations
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

DDHR_SK_Labels = {
    0: 'buildings',
    1: 'roads',
    2: 'greenery',
    3: 'water',
    4: 'farmland'
}

YESeg_Labels = {
    0: 'background',
    1: 'bare ground',
    2: 'low vegetation',
    3: 'trees',
    4: 'houses',
    5: 'water',
    6: 'roads',
    7: 'other'
}

YESeg_Mapper = np.array([255, 0, 1, 2, 3, 4, 5, 6])

PIE_Labels = {
    0: 'back broad',
    1: 'city',
    2: 'road',
    3: 'water',
    4: 'forest',
    5: 'farmland',
}

WHU_Labels = {
    0: 'background',
    1: 'farmland',
    2: 'city',
    3: 'village',
    4: 'water',
    5: 'forest',
    6: 'road',
    7: 'others'
}

WHU_Mapper = np.array([255, 0, 1, 2, 3, 4, 5, 6])

DFC_map = {
    0: "Forest", # -> 0
    1: "Shrubland", # -> 1
    2: "Grassland", # -> 2
    3: "Wetlands", # -> 3
    4: "Croplands", # -> 4
    5: "Urban/Built-up", # -> 5
    6: "Barren", # -> 6
    7: "Water", # -> 7
}

BEN_map = {
    0: "Urban fabric",
    1: "Industrial or commercial units",
    2: "Arable land",
    3: "Permanent crops",
    4: "Pastures",
    5: "Complex cultivation patterns",
    6: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    7: "Agro-forestry areas",
    8: "Broad-leaved forest",
    9: "Coniferous forest",
    10: "Mixed forest",
    11: "Natural grassland and sparsely vegetated areas",
    12: "Moors, heathland and sclerophyllous vegetation",
    13: "Transitional woodland, shrub",
    14: "Beaches, dunes, sands",
    15: "Inland wetlands",
    16: "Coastal wetlands",
    17: "Inland waters",
    18: "Marine waters",
}

EuroSat_map = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}

root_dir = {
    'DDHR-SK': r'G:\SAR_RGB_Datasets\DDHR-SK',
    'YESeg': r'G:\SAR_RGB_Datasets\YESeg-OPT-SAR',
    'PIE': r'G:\SAR_RGB_Datasets\PIE-OPT-SAR',
    'WHU': r'G:\SAR_RGB_Datasets\WHU-OPT-SAR-256',
    'DFC20': r'G:\SAR_RGB_Datasets\DFC20_MS_SAR',
    'BEN': r'G:\SAR_RGB_Datasets\BigEarthNet-MM',
    'EuroSat': r'G:\SAR_RGB_Datasets\EuroSat_MS_SAR'
}

num_labels = {
    "PIE": 6,
    "YESeg": 7,
    "DDHR-SK": 5,
    "WHU": 7,
    "DFC20": 8,
    "BEN": 19,
    "EuroSat": 10}

stats = {
    "PIE": {
            "sar": ((0.3328, 0.3328, 0.3328), (0.2265, 0.2265, 0.2265)),
            "rgb": ((0.3270, 0.3519, 0.2879), (0.2293, 0.2054, 0.2004))},
    "YESeg": {
            "sar": ((0.1598, 0.1598, 0.1598), (0.1344, 0.1344, 0.1344)),
            "rgb": ((0.3878, 0.3814, 0.3735), (0.1493, 0.1443, 0.1418))},
    "DDHR-SK": {
            "sar": ((0.1991, 0.1991, 0.1991), (0.1945, 0.1945, 0.1945)),
            "rgb": ((0.6372, 0.6446, 0.6541), (0.1629, 0.1535, 0.1451))},
    "WHU": {
            "sar": ((0.2119, 0.2119, 0.2119), (0.1911, 0.1911, 0.1911)),
            "rgb": ((0.1639, 0.1527, 0.1260), (0.0723, 0.0740, 0.0703))},
    "DFC20": {
            "sar": ((0.4864, 0.5285, 0.5074), (0.2291, 0.2357, 0.2142)),
            "rgb": ((0.2771, 0.1706, 0.2082, 0.1887, 0.2945, 0.3984, 0.4011, 0.3742, 0.4167, 0.465, 0.4689, 0.3689, 0.2852),
                    (0.2016, 0.157,  0.1618, 0.1618, 0.174, 0.1868, 0.186,  0.1832, 0.1908, 0.2124, 0.1753, 0.2023, 0.1913))},
    "BEN": {
            "sar": ((0.5521, 0.5236, 0.5378), (0.2188, 0.2209, 0.1878)),
            "rgb": ((0.296, 0.209, 0.2554, 0.2195, 0.3363, 0.3998, 0.3958, 0.3877, 0.4084, 0.4082, 0.3908, 0.3231),
                    (0.2148, 0.1666, 0.1685, 0.1781, 0.1987, 0.2136, 0.211,  0.2047, 0.2149, 0.2414, 0.2178, 0.2089))},
    "EuroSat": {
            "sar": ((0.4882, 0.5376, 0.5129), (0.2265, 0.2239, 0.1918)),
            "rgb": ((0.3945, 0.2774, 0.3149, 0.2873, 0.3832, 0.4259, 0.4229, 0.4125, 0.4312, 0.4543, 0.4891, 0.4178, 0.3789),
                    (0.2435, 0.2005, 0.2052, 0.2131, 0.2274, 0.2339, 0.2342, 0.2228, 0.2358, 0.2545, 0.2224, 0.2388, 0.2325))}}


# custom load targets and load samples. referring bigearthnet
# basedataset avilable for whu-opt-sar, yeseg-opt-sar, pie-rgb-sar, ddhr-sk
class basedataset(Dataset):
    def __init__(
            self,
            data: str,
            split: str = "train", #'train', 'val', 'test'
            img_size: int = 224,
            data_suffix: str = '.png',
            label_suffix: str = '.png',
            transforms = None,
            subset = None,
            num_class = 10,
            band = 'RGB',
            target_size = 224,
            seed=0,
    ):
        self.data = data
        self.stats = stats[data]
        self.target_size = target_size
        self.root_dir = root_dir[data]
        self.split = split
        self.img_size = img_size
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.transforms = transforms
        self.band = band
        self.num_class = num_class
        if data == 'YESeg':
            self.label_map = YESeg_Mapper
        elif data == 'WHU':
            self.label_map = WHU_Mapper
        else:
            self.label_map = None
        
        self.samples = []
        self._collect_samples()
        if subset is not None:
            rng = np.random.RandomState(729+seed)
            rng.shuffle(self.samples)
            if subset < 1.:
                subset_len = int(len(self.samples) * subset)
            else:
                subset_len = int(subset * self.num_class)
            self.samples = self.samples[:subset_len]
    
    def _collect_samples(self):
        split_file = os.path.join(self.root_dir, self.split+'.txt')
        with open(split_file, 'r') as f:
            for line in f:
                img_name = os.path.splitext(line.strip())[0]
                rgb_path = os.path.join(self.root_dir, "RGB", img_name)
                sar_path = os.path.join(self.root_dir, "SAR", img_name)
                label_path = os.path.join(self.root_dir, "Label", img_name)
                self.samples.append({
                    "sar_path": sar_path,
                    "rgb_path": rgb_path,
                    "label_path": label_path})
    
    def _load_samples(self, path):
        if self.data_suffix == '.png':
            loaded = Image.open(path + self.data_suffix)
            if loaded.mode != 'RGB':
                loaded = loaded.convert('RGB')
            img = np.array(loaded)
        elif self.data_suffix == '.npy':
            img = np.load(path + self.data_suffix)
        else:
            raise NotImplementedError
        return img
    
    def _load_targets(self, path):
        if self.label_suffix == '.png':
            label_img = np.array(Image.open(path + self.label_suffix))
            if self.label_map is not None:
                label_img = self.label_map[label_img]
            unique, counts = np.unique(label_img, return_counts=True)
            indices = [class_idx for class_idx, num in zip(unique, counts) if num/label_img.shape[1]**2 >= 0.1 and class_idx != 255]
            label = torch.zeros(self.num_class, dtype=torch.long)
            for i in indices:
                label[i] = 1
        elif self.label_suffix == '.npy':
            label = torch.from_numpy(np.load(path + self.label_suffix)).long()
        else:
            raise NotImplementedError
        return label
    
    def _normalize(self, rgb_img, sar_img):
        rgb_mean, rgb_std = self.stats['rgb']
        mean_tensor = torch.tensor(rgb_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(rgb_std).view(-1, 1, 1)
        rgb_img = (rgb_img - mean_tensor) / std_tensor
        
        sar_mean, sar_std = self.stats['sar']
        mean_tensor = torch.tensor(sar_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(sar_std).view(-1, 1, 1)
        sar_img = (sar_img - mean_tensor) / std_tensor
        return rgb_img, sar_img
    
    def check_chs(self, rgb_img, sar_img):
        if self.band != 'RGB':
            sar_img = sar_img[:2,:,:]
            _rgb_img = torch.zeros([13 if self.band == 'B13' else 12, rgb_img.shape[1], rgb_img.shape[2]], device=rgb_img.device, dtype=rgb_img.dtype)
            
            _rgb_img[:] = rgb_img.mean(dim=0, keepdim=True)
            
            _rgb_img[[3,2,1]] = rgb_img
            rgb_img = _rgb_img
        return rgb_img, sar_img
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx):
        #print(idx)
        sample = self.samples[idx]
        
        rgb_img = self._load_samples(sample["rgb_path"])
        sar_img = self._load_samples(sample["sar_path"])
        label = self._load_targets(sample["label_path"])
        
        if self.transforms is not None:
            transformed = self.transforms(image=np.array(sar_img), rgb=np.array(rgb_img))
            rgb_img = transformed["rgb"]
            sar_img = transformed["image"]
            rgb_img, sar_img = self._normalize(rgb_img/255, sar_img/255)
            
        rgb_img, sar_img = self.check_chs(rgb_img, sar_img)
    
        return {'rgb': rgb_img, 'sar': sar_img, 'label': label}

class DFC20dataset(basedataset):
    def __init__(self, **kwargs):  # band = 'RGB', 'B12', or 'B13'
        super().__init__(**kwargs)
    
    # DFC20 bands: [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12] & [VV, VH]
    def _load_samples(self, path):
        img = super()._load_samples(path)
        modality = os.path.split(os.path.split(path)[0])[-1]
        if modality == 'RGB':
            if self.band == 'RGB':
                img = img[:, :, [3, 2, 1]]
            elif self.band == 'B12':
                img = img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
            elif self.band == 'B13':
                img = img
        elif modality == 'SAR':
            if self.band == 'RGB':
                average_channel = np.mean(img, axis=2)
                img = np.concatenate([img, average_channel[:, :, np.newaxis].astype(np.uint8)], axis=2)
        return img
    
    def _normalize(self, rgb_img, sar_img):
        rgb_mean, rgb_std = self.stats['rgb']
        mean_tensor = torch.tensor(rgb_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(rgb_std).view(-1, 1, 1)
        if self.band == 'RGB':
            mean_tensor, std_tensor = mean_tensor[[3,2,1]], std_tensor[[3,2,1]]
        elif self.band == 'B12':
            mean_tensor, std_tensor = mean_tensor[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]], std_tensor[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]
        rgb_img = (rgb_img - mean_tensor) / std_tensor
        
        sar_mean, sar_std = self.stats['sar']
        mean_tensor = torch.tensor(sar_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(sar_std).view(-1, 1, 1)
        if self.band != 'RGB':
            mean_tensor, std_tensor = mean_tensor[:2], std_tensor[:2]
        sar_img = (sar_img - mean_tensor) / std_tensor
        return rgb_img, sar_img
    
    def check_chs(self, rgb_img, sar_img):
        return rgb_img, sar_img

class BENdataset(basedataset):
    def __init__(self, **kwargs):  # band = 'RGB', 'B12', 'B13'
        super().__init__(**kwargs)
    
    # BEN bands: [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] & [VV, VH]
    def _load_samples(self, path):
        img = super()._load_samples(path)
        modality = os.path.split(os.path.split(path)[0])[-1]
        if modality == 'RGB':
            if self.band == 'RGB':
                img = img[:, :, [3, 2, 1]]
            elif self.band == 'B13':
                nullband = np.zeros_like(img[:, :, [0]])
                img = np.concatenate([img[:,:,:10], nullband, img[:,:,10:]], axis=2)
        elif modality == 'SAR':
            if self.band == 'RGB':
                average_channel = np.mean(img, axis=2)
                img = np.concatenate([img, average_channel[:, :, np.newaxis].astype(np.uint8)], axis=2)
        return img
    
    def _normalize(self, rgb_img, sar_img):
        rgb_mean, rgb_std = self.stats['rgb']
        mean_tensor = torch.tensor(rgb_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(rgb_std).view(-1, 1, 1)
        if self.band == 'RGB':
            mean_tensor, std_tensor = mean_tensor[[3, 2, 1]], std_tensor[[3, 2, 1]]
        elif self.band == 'B13':
            null_band = torch.zeros_like(mean_tensor[0:1])
            mean_tensor = torch.cat([mean_tensor[:10], null_band, mean_tensor[10:]], dim=0)
            std_tensor = torch.cat([std_tensor[:10], null_band+1., std_tensor[10:]], dim=0)
        rgb_img = (rgb_img - mean_tensor) / std_tensor
        
        sar_mean, sar_std = self.stats['sar']
        mean_tensor = torch.tensor(sar_mean).view(-1, 1, 1)
        std_tensor = torch.tensor(sar_std).view(-1, 1, 1)
        if self.band != 'RGB':
            mean_tensor, std_tensor = mean_tensor[:2], std_tensor[:2]
        sar_img = (sar_img - mean_tensor) / std_tensor
        return rgb_img, sar_img
    
    def check_chs(self, rgb_img, sar_img):
        return rgb_img, sar_img

class EuroSatMMdataset(DFC20dataset):
    def __init__(self,  **kwargs):  # band = 'RGB', 'B12', 'B13'
        self.label_index = {v: k for k, v in EuroSat_map.items()}
        super().__init__(**kwargs)
    
    # EuroSat bands: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'] & [VV, VH]  Note: 'B8A' re-located at pos 8 at preprocessing
    def _load_targets(self, path):
        class_name = os.path.basename(path).split('_')[0]
        class_idx = self.label_index[class_name]
        label = torch.tensor(class_idx).long()
        # label = torch.zeros(self.num_class, dtype=torch.long)
        # label[class_idx] = 1
        return label


def get_dataloader(args, generator):
    
    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        worker_seed = torch.initial_seed() % 2 ** 32 + args.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    if not args.linprob:
        train_transform = A.Compose([
            A.RandomResizedCrop(
                size=(args.input_size, args.input_size),
                scale=(0.8, 1.0),
                interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()], seed=27, additional_targets={'rgb': 'image'}, save_applied_params=True)
    else:
        train_transform = A.Compose([
            A.Resize(
                height=args.input_size,
                width=args.input_size,
                interpolation=cv2.INTER_CUBIC),
            ToTensorV2()], additional_targets={'rgb': 'image'})
        
        
    val_transform = A.Compose([
        A.Resize(args.input_size, args.input_size, interpolation=cv2.INTER_CUBIC),
        ToTensorV2()], additional_targets={'rgb': 'image'})
    
    
    num_classes = num_labels[args.dataset]
    
    if args.dataset != 'DFC20' and args.dataset != 'BEN' and args.dataset != 'EuroSat':
        train_dataset = basedataset(data=args.dataset,
                                    split="train",
                                    data_suffix='.png',
                                    label_suffix='.png',
                                    band = args.bands,
                                    transforms=train_transform,
                                    num_class=num_classes,
                                    subset=args.subset,
                                    target_size=args.input_size,
                                    seed=args.seed)
        
        val_dataset = basedataset(data=args.dataset,
                                    split="test",
                                    data_suffix='.png',
                                    label_suffix='.png',
                                    band=args.bands,
                                    transforms=val_transform,
                                    num_class=num_classes,
                                    target_size=args.input_size)
    elif args.dataset == 'DFC20':
        train_dataset = DFC20dataset(data='DFC20',
                                     split="test",
                                     band=args.bands,  # 'RGB' 'B12' 'B13'
                                     data_suffix='.npy',
                                     label_suffix='.png',
                                     transforms=train_transform,
                                     num_class=num_classes,
                                     subset=args.subset,
                                     target_size=args.input_size,
                                     seed=args.seed)
        
        val_dataset = DFC20dataset(data='DFC20',
                                     split="val",
                                     band=args.bands,  # 'RGB' 'B12' 'B13'
                                     data_suffix='.npy',
                                     label_suffix='.png',
                                     transforms=val_transform,
                                     num_class=num_classes,
                                     target_size=args.input_size)
    elif args.dataset == 'BEN':
        train_dataset = BENdataset(data='BEN',
                                   split="train",
                                   band=args.bands,  #'RGB' 'B12'
                                   data_suffix='.npy',
                                   label_suffix='.npy',
                                   transforms=train_transform,
                                   num_class=num_classes,
                                   subset=args.subset,
                                   target_size=args.input_size,
                                   seed=args.seed)
        
        val_dataset = BENdataset(data='BEN',
                                   split="val",
                                   band=args.bands,  #'RGB' 'B12'
                                   data_suffix='.npy',
                                   label_suffix='.npy',
                                   transforms=val_transform,
                                   num_class=num_classes,
                                   target_size=args.input_size)
    elif args.dataset == 'EuroSat':
        train_dataset = EuroSatMMdataset(data='EuroSat',
                                         split="train",
                                         band=args.bands,   #'RGB' 'B12' 'B13'
                                         data_suffix='.npy',
                                         label_suffix='.npy',
                                         transforms=train_transform,
                                         num_class=num_classes,
                                         subset=args.subset,
                                         target_size=args.input_size,
                                         seed=args.seed)
        
        val_dataset = EuroSatMMdataset(data='EuroSat',
                                       split="test",
                                       band=args.bands,  # 'RGB' 'B12' 'B13'
                                       data_suffix='.npy',
                                       label_suffix='.npy',
                                       transforms=val_transform,
                                       num_class=num_classes,
                                       target_size=args.input_size)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.bs,
                              shuffle=True,
                              drop_last=True if len(train_dataset) > args.bs else False,
                              generator=generator,
                              worker_init_fn=seed_worker,
                              num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=args.valbs, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, num_classes

if __name__ == '__main__':
    img = np.load(r'G:\SAR_RGB_Datasets\BigEarthNet-MM\RGB\S2A_MSIL2A_20170613T101031_0_48_S1B_IW_GRDH_1SDV_20170613T160455_34VER_0_48.npy')
    print()
