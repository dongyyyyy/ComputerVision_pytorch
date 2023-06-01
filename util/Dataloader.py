from torch.utils.data import Dataset

import torch

# pip install python-opencv
import cv2
# pip install pandas
import pandas as pd
# pip install numpy
import numpy as np

import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from util.config import *
from ImageNet_classes import *
from util.function import onehot,rand_bbox


class ImageNet_dataloader(Dataset):
    
    
    def __init__(self, path, input_size,num_class=1000,cutmix_p = 0.,beta = 1.,training=False):
        self.image_path = path
        self.input_size = input_size
        
        # for one-hot
        self.num_class = num_class
        
        #for cutmix
        self.cutmix_p = cutmix_p
        self.beta = beta
        if training:
            self.transform_wihtcutout = A.Compose(
                                        [
                                            A.Resize(input_size+32,input_size+32),
                                            A.RandomCrop(input_size,input_size),
                                            A.OneOf([
                                                A.HorizontalFlip(p=1),
                                                A.RandomRotate90(p=1),
                                                A.VerticalFlip(p=1)
                                            ],p=0.9),
                                            # 적용됐을때 이미지 예시 만들기 !
                                            A.OneOf([
                                                A.RandomBrightnessContrast(p=1),  # Control the Brightness
                                                # A.MotionBlur(p=1), # Blur(Like Moving)
                                                A.GaussianBlur(p=1),
                                                A.OpticalDistortion(p=1), # 윤곽 왜곡
                                                A.GaussNoise(p=1) # 가우시난 노이즈
                                            ],p=0.9),
                                            A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.8), # Cutout
                                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalization using ImageNet mean and std
                                            # A.pytorch.ToTensor()
                                            # A.pytorch.ToTensorV2(transpose_mask=True)
                                            ToTensorV2(transpose_mask=True)
                                        ]
                                    )
            self.transform_wihtoutcutout = A.Compose(
                                        [
                                            A.Resize(input_size+32,input_size+32),
                                            A.RandomCrop(input_size,input_size),
                                            A.OneOf([
                                                A.HorizontalFlip(p=1),
                                                A.RandomRotate90(p=1),
                                                A.VerticalFlip(p=1)
                                            ],p=0.9),
                                            # 적용됐을때 이미지 예시 만들기 !
                                            A.OneOf([
                                                A.RandomBrightnessContrast(p=1),  # Control the Brightness
                                                A.MotionBlur(p=1), # Blur(Like Moving)
                                                A.GaussianBlur(p=1),
                                                A.OpticalDistortion(p=1), # 윤곽 왜곡
                                                A.GaussNoise(p=1) # 가우시난 노이즈
                                            ],p=0.9),
                                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Normalization using ImageNet mean and std
                                            # A.pytorch.ToTensor()
                                            # A.pytorch.ToTensorV2(transpose_mask=True)
                                            ToTensorV2(transpose_mask=True)
                                        ]
                                    )
            #                                 A.SmallestMaxSize(max_size=input_size+60),
            #                                 A.RandomCrop(height=input_size, width=input_size),
            #                                 A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            #                                 A.RandomBrightnessContrast(p=0.5),
            #                                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #                                 ToTensorV2(),
        else:
            val_transform = A.Compose(
                                        [
                                            A.Resize(input_size,input_size),
                                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                            # A.pytorch.ToTensor()
                                            ToTensorV2(transpose_mask=True)
                                            # A.pytorch.ToTensorV2(transpose_mask=True)
                                            # ToTensorV2(),
                                        ]
                                     )
        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, id: int):
        # class_num,class_name = class_information[self.image_path[id].split('/')[-2]]
        class_num,class_name = ImageNet_class_dict[self.image_path[id].split('/')[-2]]
        image = cv2.imread(self.image_path[id], cv2.IMREAD_COLOR)
        orig_size = image.shape

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.cutmix_p > 0.:
            augmented = self.transform_wihtoutcutout(image=image)
        else:
            augmented = self.transform_wihtcutout(image=image)
        
        image = augmented['image']
        if self.cutmix_p > 0.:
            while(1):
                target_index = np.random.randint(0,self.__len__())
                if target_index != id: # find different image!
                    break
            class_num_target,class_name_target = ImageNet_class_dict[self.image_path[target_index].split('/')[-2]]
            image_target = cv2.imread(self.image_path[target_index],cv2.IMREAD_COLOR)
            image_target = cv2.cvtColor(image_target,cv2.COLOR_BGR2RGB)
            augmented_target = self.transform_wihtoutcutout(image=image_target)
            image_target = augmented_target['image']
            
            # ratio of mixing
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            label_onehot_source = onehot(self.num_class, class_num)
            label_onehot_target = onehot(self.num_class, class_num_target)

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, bbx1:bbx2, bby1:bby2] = image_target[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            label_onethot = label_onehot_source * lam + label_onehot_target * (1. - lam)
            
            return {'image':image,'label':label_onethot,'class_name':[class_name,class_name_target]}
        # print(image.shape)
        return {'image':image,'label':int(class_num),'class_name':class_name}
        

# Normalized Tensor to de-normalized Tensor
class UnNormalize(object):
    def __init__(self, mean=torch.tensor((0.485, 0.456, 0.406)), std=torch.tensor((0.229, 0.224, 0.225))):
        self.mean = mean.reshape(-1,1,1)
        self.std = std.reshape(-1,1,1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor * self.std + self.mean
            
def show_image_example(num_samples = 5):
    fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
    for i in range(num_samples):
        ax[i].imshow(transforms.ToPILImage()(unNormalization_method(albumentations_dataset[0]['image'])))
        ax[i].axis('off')