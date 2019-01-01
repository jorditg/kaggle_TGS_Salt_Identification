import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Augment(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img, msk):
        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            img = transforms.functional.hflip(img)
            msk = transforms.functional.hflip(msk)

        p = np.random.uniform(0, 1, 1)
        if p > 0.5:
            img = transforms.functional.vflip(img)
            msk = transforms.functional.vflip(msk)

        # convert to tensor        
        img = transforms.functional.to_tensor(img)
        msk = transforms.functional.to_tensor(msk)

        # only 1 channel
        img = img[0,:,:].unsqueeze(0)
        msk = img[0,:,:].unsqueeze(0)
        
        # normalize image
        img  = (img - self.mean)/self.std

        return img, msk

class TGSDataset(Dataset):
    def __init__(self, image_paths, target_paths, mean = 0.0, stddev = 1.0):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.augment = Augment(mean, stddev)
        
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        target = Image.open(self.target_paths[index])

        image, target = self.augment(image, target)
        return image, target
    
    def __len__(self):                                                                               
        return len(self.image_paths) 
                                       
