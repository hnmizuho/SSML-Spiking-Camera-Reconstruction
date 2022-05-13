#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from utils import *

def load_dataset(root_dir, redux, params, crop_size,shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    if params.net == "csy":
        dataset = SpikeDataset(root_dir, redux, crop_size,
            clean_targets=params.clean_targets)  

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled,num_workers=20,pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled,num_workers=20,pin_memory=True)


class AbstractDataset(Dataset):

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets
        # self.transform = transforms.Compose([transforms.RandomCrop(self.crop_size),])
        
    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w,h=400,250
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            img = tvF.to_pil_image(img) 
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            img = tvF.crop(img, i, j, self.crop_size, self.crop_size)
            img = tvF.to_tensor(img)
            cropped_imgs.append(img)

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)

class SpikeDataset(AbstractDataset):

    def __init__(self, root_dir, redux, crop_size, clean_targets=False):
        """Initializes noisy image dataset."""

        super(SpikeDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs_pair1 = os.listdir(root_dir)
        self.imgs_pair1 = list(set(self.imgs_pair1) - set([".ipynb_checkpoints"]))
        self.imgs_pair1 = sorted(self.imgs_pair1) #800
        if redux:
            self.imgs_pair1 = self.imgs_pair1[:redux]
    def __getitem__(self, index):
        """Retrieves image from data folder."""

        spk_path = os.path.join(self.root_dir, self.imgs_pair1[index])
        spk = load_vidar_dat(spk_path,frame_cnt=41) #41 1 250 400

        tfi = []
        for i in [6,13,20,27,34]:
            tmp = middleTFI(spk, i, window=12) #250 400
            tmp = torch.tensor(tmp,dtype=torch.float32).unsqueeze_(dim=0) #1 250 400
            tfi.append(tmp)
#             tmp2=torch.mean(spk[i-5:i+5+1,:,:,:],dim=0) / 0.6 #1 250 400
#             tfi.append(tmp2)
        tfi = torch.stack(tfi,dim=0) #5 1 250 400        
        
        gt = copy.deepcopy(tfi)
            
        if self.crop_size != 0:
            tmp = torch.cat([spk,tfi,gt])#41+5+5 1 250 400
            tmp = self._random_crop(tmp)#51 1 250 400
            tmp = torch.stack(tmp,dim=0)#51 1 40 40
            tmp = tmp.squeeze_(1).numpy()       #51 40 40    
            mode = np.random.choice([0,1,2,3,4,5,6,7],1,replace=False)  
            tmp = data_augmentation(tmp,mode)
            tmp = torch.from_numpy(tmp.copy()) #51 40 40    
            spk = tmp[:41]
            tfi = tmp[41:41+5]
            gt  = tmp[41+5:41+5+5]
        else:
            spk = spk.squeeze_(dim=1) #20 1 250 400 --> 20 250 400
            tfi = tfi.squeeze_(dim=1)
            gt  = gt.squeeze_(dim=1)

        return spk,tfi,gt # source target gt

    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs_pair1)
 