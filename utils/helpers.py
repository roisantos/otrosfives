import os
import pickle
import random
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore



def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def read_pickle(path, type):
    with open(file=path + f"/{type}.pkl", mode='rb') as file:
        img = pickle.load(file)
    return img


def save_pickle(path, type, img_list):
    with open(file=path + f"/{type}.pkl", mode='wb') as file:
        pickle.dump(img_list, file)

# https://github.com/ykwon0407/UQ_BNN/blob/master/retina/utils.py (Keras framework)
def random_snippet(x, y, size=(48,48), rotate=True, flip=True):
    
    '''sample snippets from images. return image tuple (real, segmentation) of size `size` '''
    
    assert x.shape[:2] == y.shape[:2]
    
    # get image sample
    sample = np.random.randint(0, x.shape[0])
    # get x and y dimensions
    min_h = np.random.randint(0, x.shape[1]-size[0])
    max_h = min_h+size[0]
    min_w = np.random.randint(0, x.shape[2]-size[1])
    max_w = min_w+size[1]
    # extract snippet
    im_x = x[sample, min_h:max_h, min_w:max_w, :]
    im_y = y[sample, min_h:max_h, min_w:max_w, :]
    
    # rotate
    if rotate:
        k = np.random.randint(0,4)
        im_x = np.rot90(im_x, k=k, axes=(0,1))
        im_y = np.rot90(im_y, k=k, axes=(0,1))
        
    # flip left-right, up-down
    if flip:
        if np.random.random() < 0.5:
            lr_ud = np.random.randint(0,2) # flip up-down or left-right?
            im_x = np.flip(im_x, axis=lr_ud)
            im_y = np.flip(im_y, axis=lr_ud)
    
    return (im_x, im_y)

def get_random_snippets(x, y, number, size):
    snippets = [random_snippet(x, y, size) for i in range(number) ]
     
    ims_x = np.array([i[0] for i in snippets])
    ims_y = np.array([i[1] for i in snippets])
    return (ims_x, ims_y)   


def double_threshold_iteration(index,img, h_thresh, l_thresh, save=True):
    h, w = img.shape
    img = np.array(torch.sigmoid(img).cpu().detach()*255, dtype=np.uint8)
    bin = np.where(img >= h_thresh*255, 255, 0).astype(np.uint8)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                if gbin[i][j] == 0 and img[i][j] < h_thresh*255 and img[i][j] >= l_thresh*255:
                    if gbin[i-1][j-1] or gbin[i-1][j] or gbin[i-1][j+1] or gbin[i][j-1] or gbin[i][j+1] or gbin[i+1][j-1] or gbin[i+1][j] or gbin[i+1][j+1]:
                        gbin[i][j] = 255

    if save:
        cv2.imwrite(f"save_picture/bin{index}.png", bin)
        cv2.imwrite(f"save_picture/gbin{index}.png", gbin)
    return gbin/255
