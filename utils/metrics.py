import numpy as np
import cv2
import os
from glob import glob
from loguru import logger

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


# CLAHE
def clahe_equalized(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr

def fives_loader(Dataset, CFG):

    # Split dataset into train and validation
    validation_split = .2  # Hardcoded split, as in the original
    shuffle_dataset = True

    # Creating data indices for training and validation splits:
    dataset_size = len(Dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(CFG['random_seed'])
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Return DataLoader instances, not just samplers
    train_loader = DataLoader(Dataset, batch_size=CFG['batch_size'], pin_memory=True,
                              sampler=train_sampler, drop_last=True, num_workers=CFG['num_workers'])
    val_loader = DataLoader(Dataset, batch_size=CFG['batch_size'], drop_last=True,
                            sampler=valid_sampler, pin_memory=True, num_workers=CFG['num_workers'])

    logger.info(
        'The total number of images for train and validation is %d' % len(Dataset))

    return train_loader, val_loader  # Return DataLoaders

def fives_test_loader(Dataset):

    loader = DataLoader(dataset=Dataset, batch_size=1,
                        shuffle=False, pin_memory=True, num_workers=8)

    return loader

def load_subgroup_images(disease, root):
    def load_paths(subdir):
        # Use os.path.join for proper path construction
        original = sorted(glob(os.path.join(root, subdir, 'Original/*')))
        ground_truth = sorted(glob(os.path.join(root, subdir, 'Ground truth/*')))

        # Exclude the database file from training images only
        if 'train' in subdir:
            original = original[:-1]

        return original, ground_truth

    train_x, train_y = load_paths('train')
    valid_x, valid_y = load_paths('test')

    # Split into training and validation sets based on the presence of 'disease' in the filename
    def split_data(items):
        return ([item for item in items if disease in os.path.basename(item)],
                [item for item in items if disease not in os.path.basename(item)])

    valid_x, train_x = split_data(train_x + valid_x)
    valid_y, train_y = split_data(train_y + valid_y)

    return train_x, train_y, valid_x, valid_y

import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, cohen_kappa_score, matthews_corrcoef
import numpy as np
import cv2

# code: https://github.com/lseventeen/FR-UNet/blob/master/utils/metrics.py

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return np.round(self.val, 4) # type: ignore

    @property
    def average(self):
        return np.round(self.avg, 4) # type: ignore


def get_metrics(predict, target, threshold=None, predict_b=None):
    epsilon = 1e-8  # Add a small epsilon value
    predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
    if predict_b is not None:
        predict_b = predict_b.flatten()
    else:
        predict_b = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()

    # --- Fix: Check for single class in target ---
    if np.unique(target).size == 1:
        auc = 0.0  # Return a default value (e.g., 0)
    else:
        auc = roc_auc_score(target.astype(np.uint8), predict)
    # --- End Fix ---
        
    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    #auc = roc_auc_score(target.astype(np.uint8), predict)
    # auc = 0.000
    # auc = roc_auc_score(target, predict)
    mcc = matthews_corrcoef(target.astype(int), predict_b.astype(int))
    acc = (tp + tn + epsilon) / (tp + fp + fn + tn + epsilon)
    pre = (tp + epsilon) / (tp + fp + epsilon)
    sen = (tp + epsilon) / (tp + fn + epsilon)
    spe = (tn + epsilon) / (tn + fp + epsilon)
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    f1 = (2 * pre * sen + epsilon) / (pre + sen + epsilon)
    
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
        "MCC": np.round(mcc, 4)
    }


def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n
