import argparse
import os

import imageio
import numpy as np
from torch.utils.data import Dataset
# import cv2
import imutils
import matplotlib.pyplot as plt
from torch.utils import data
from PIL import Image
import albumentations as A

def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader,
                 suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type
        self.suffix = suffix
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size
        
        self.colojitter = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8)
    
    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_scale(pre_img, post_img, label, [0.5, 1.5])
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
            pre_img, post_img = self.colojitter(image=pre_img.astype(np.uint8))['image'], self.colojitter(image=post_img.astype(np.uint8))['image']
        
        # pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = imutils.normalize_img(pre_img, mean=[88.76, 85.20, 76.74], std=[67.49, 58.56, 54.09])
        pre_img = np.transpose(pre_img, (2, 0, 1))
        
        # post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = imutils.normalize_img(post_img, mean=[56.29, 56.29, 56.29], std=[45.01, 45.01, 45.01])
        post_img = np.transpose(post_img, (2, 0, 1))
        
        return pre_img, post_img, label
    
    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event', self.data_list[index] + '_post_disaster' + self.suffix)
        label_path = os.path.join(self.dataset_path, 'target', self.data_list[index] + '_building_damage' + self.suffix)
        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,) * 3, axis=-1)
        clf_label = self.loader(label_path)
        
        if 'train' in self.data_pro_type:
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
            clf_label = np.asarray(clf_label)
        loc_label = clf_label.copy()
        loc_label[loc_label == 2] = 1
        loc_label[loc_label == 3] = 1
        
        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, data_idx
    
    def __len__(self):
        return len(self.data_list)


class MultimodalDamageAssessmentDatset_Inference(Dataset):
    def __init__(self, dataset_path, data_list, data_loader=img_loader, suffix='.tif'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.suffix = suffix
    
    def __transforms(self, pre_img, post_img):
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))
        
        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))
        
        return pre_img, post_img
    
    def __getitem__(self, index):
        pre_path = os.path.join(self.dataset_path, 'pre-event', self.data_list[index] + '_pre_disaster' + self.suffix)
        post_path = os.path.join(self.dataset_path, 'post-event',
                                 self.data_list[index] + '_post_disaster' + self.suffix)
        pre_img = self.loader(pre_path)[:, :, 0:3]
        post_img = self.loader(post_path)
        
        # pre_img = np.stack((pre_img,)*3, axis=-1)
        post_img = np.stack((post_img,) * 3, axis=-1)
        
        pre_img, post_img = self.__transforms(pre_img, post_img)
        
        data_idx = self.data_list[index]
        return pre_img, post_img, data_idx
    
    def __len__(self):
        return len(self.data_list)