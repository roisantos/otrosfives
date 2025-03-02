import os
import numpy as np
import pickle
import cv2
import torch
from torch.utils.data import Dataset


from PIL import Image
from glob import glob
from datasets.utils import clahe_equalized
from datasets.transform import pipeline_tranforms


class FIVES(Dataset):

    def __init__(self, CFG, mode='train'):  # Remove indices, add mode
        super(FIVES, self).__init__()
        self.mode = mode
        self.transforms = pipeline_tranforms()
        self.CFG = CFG

        # Use relative paths based on the project root and mode
        if mode == 'train':
            self.images_path = sorted(glob(os.path.join(CFG['dataset']['path'], "train", "image", "*")))
            self.masks_path = sorted(glob(os.path.join(CFG['dataset']['path'], "train", "label", "*")))
        elif mode == 'test':
            self.images_path = sorted(glob(os.path.join(CFG['dataset']['path'], "test", "image", "*")))
            self.masks_path = sorted(glob(os.path.join(CFG['dataset']['path'], "test", "label", "*")))
        # No 'val' mode needed; splitting is done in fives_loader

        self.n_samples = len(self.images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = clahe_equalized(image)
        image = cv2.resize(image, (self.CFG['size'], self.CFG['size']), interpolation=cv2.INTER_NEAREST)

        image = image / 255.0  # type: ignore # (512, 512, 3) Normalizing to range (0,1)
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,  (self.CFG['size'], self.CFG['size']), interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0  # type: ignore # (512, 512)
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        # common transform
        if self.mode == 'train':
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transforms(image) # type: ignore
            torch.manual_seed(seed)
            mask = self.transforms(mask) # type: ignore

        return image, mask

    def __len__(self):
        return self.n_samples
