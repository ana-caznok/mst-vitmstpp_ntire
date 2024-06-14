import os 
import logging
import datetime
import hdf5storage
import h5py 
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import torch
from torchmetrics import StructuralSimilarityIndexMeasure

def initiate_logger(file_dir, name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[(%(asctime)s] %(message)s")

    handler = logging.FileHandler(file_dir)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    

    logger.addHandler(handler)

    return logger 


def create_folders_for(saved_dir, logged_dir, model_name = "", external_dir = None):
    experiment_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if external_dir is not None:
        os.mkdir(external_dir)
        saved_dir = f"{external_dir}/{saved_dir}"
        logged_dir = f"{external_dir}/{logged_dir}"
        reconstructed_dir = f"{external_dir}/{reconstructed_dir}"

    if (not os.path.exists(logged_dir)):
        os.mkdir(logged_dir)

    if (not os.path.exists(saved_dir)):
        os.mkdir(saved_dir)

    return logged_dir, saved_dir

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def save_jpg(path, rgb, quality):
    rgb = rgb.clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

def loadcube(path):
    with h5py.File(path, 'r') as f:
        cube = np.squeeze(np.float32(np.array(f['cube'])))
        cube = np.transpose(cube, [2,1,0]) 
        f.close()
    return cube

def visualize_save_cube32(cube, bands, rgb_file, saved_path, MSI = False):
    plt.figure(figsize=(20, 10))
    if MSI:
        rgb = loadcube(rgb_file)[:,:,:3]
    else:
        rgb = plt.imread(rgb_file)

    plt.subplot(4,8,1), plt.imshow(rgb), plt.title('RGB')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    for k in range(2, 33):
        plt.subplot(4,8,k)
        plt.imshow(cube[k-2]), plt.title(f'Band {bands[k-2]}')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{saved_path}.png')


def visualize_save_cube63(cube, bands, rgb_file, saved_path, MSI = False):
    plt.figure(figsize=(20, 5))
    if MSI:
        rgb = loadcube(rgb_file)[:,:,:3]
    else:
        rgb = plt.imread(rgb_file)

    plt.subplot(2,31,1), plt.imshow(rgb), plt.title('RGB')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    for k in range(2, 63):
        plt.subplot(2,31,k)
        plt.imshow(cube[k-2]), plt.title(f'Band {bands[k-2]}')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{saved_path}.png')

def visualize_sam_map(sam_map, saved_path):
    sam_map = np.array(sam_map).squeeze()
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 15))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(sam_map[i, :, :], vmin=0, vmax=1, cmap='gray')
        ax.axis('off')
        ax.set_title(f"SAM: {sam_map[i, :, :].mean():.3f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), pad = 0.01)
    plt.savefig(f'{saved_path}.png')

def sam_fn(pred, target):
    '''
    pred, target: [c, w, h]
    '''
    pred, target = pred.squeeze(), target.squeeze()
    up = torch.sum((target*pred), dim = 0)   # [w, h]
    down1 = torch.sum((target**2), dim = 0).sqrt()
    down2 = torch.sum((pred**2), dim = 0).sqrt()

    map = torch.arccos(up / (down1 * down2))
    score = torch.mean(map[~torch.isnan(map)])
    map[torch.isnan(map)] = 0
    return score, map

def calc_ssim(pred,target):

    ssim = StructuralSimilarityIndexMeasure(pred, target, compute_with_cache=False)

    return ssim