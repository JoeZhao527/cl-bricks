from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import pickle
from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
import pandas as pd
import random
from zipfile import ZipFile
import os
import torch
import torch.nn.functional as F
from typing import Tuple, List
import lmdb
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import random

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def collate_fn(batch: List[torch.Tensor], target_dim: Tuple[int, int] = (128, 64)):
    """
    Collate function to generate a batch suitable for InfoNCE loss.
    It creates two augmented views for each sample by randomly cropping
    along the time dimension and padding to the target dimensions with padding
    applied only on the right side.

    Args:
        batch (List[torch.Tensor]): List of spectrogram tensors. Each tensor has shape (height, width).
        target_dim (Tuple[int, int]): Target dimensions (height, width).

    Returns:
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            - Two tensors containing augmented features (view1 and view2).
            - Labels indicating positive pairs.
    """
    augmented_features1 = []
    augmented_features2 = []
    
    target_height, target_width = target_dim

    for feat in batch:
        _, current_width = feat.shape

        # --- Augmentation View 1 ---
        # Randomly crop along the time dimension if necessary
        aug_1_width = random.randint(target_width // 3, target_width)
        if current_width > aug_1_width:
            start = random.randint(0, current_width - aug_1_width)
            feat_aug1 = feat[:, start:start + aug_1_width]
        else:
            feat_aug1 = feat.clone()
        
        # Pad on the right if necessary
        pad_width1 = max(target_width - feat_aug1.shape[1], 0)
        if pad_width1 > 0:
            pad = (0, pad_width1)  # (left, right)
            feat_aug1 = F.pad(feat_aug1, pad, mode='constant', value=0)
        
        # --- Augmentation View 2 ---
        # Another random crop along the time dimension
        aug_2_width = random.randint(target_width // 3, target_width)
        if current_width > aug_2_width:
            start = random.randint(0, current_width - aug_2_width)
            feat_aug2 = feat[:, start:start + aug_2_width]
        else:
            feat_aug2 = feat.clone()
        
        # Pad on the right if necessary
        pad_width2 = max(target_width - feat_aug2.shape[1], 0)
        if pad_width2 > 0:
            pad = (0, pad_width2)  # (left, right)
            feat_aug2 = F.pad(feat_aug2, pad, mode='constant', value=0)
        
        # --- Padding Height ---
        # Pad height on the bottom if necessary
        pad_height = max(target_height - feat_aug1.shape[0], 0)
        if pad_height > 0:
            pad = (0, 0, 0, pad_height)  # (left, right, top, bottom)
            feat_aug1 = F.pad(feat_aug1, pad, mode='constant', value=0)
            feat_aug2 = F.pad(feat_aug2, pad, mode='constant', value=0)
        else:
            # Truncate if necessary
            feat_aug1 = feat_aug1[:target_height, :]
            feat_aug2 = feat_aug2[:target_height, :]
        
        # --- Final Shape Verification ---
        assert feat_aug1.shape == target_dim, \
            f"Augmented feature1 has shape {feat_aug1.shape}, expected {target_dim}"
        assert feat_aug2.shape == target_dim, \
            f"Augmented feature2 has shape {feat_aug2.shape}, expected {target_dim}"
        
        augmented_features1.append(feat_aug1)
        augmented_features2.append(feat_aug2)
    
    # Stack augmented features into tensors
    batch_features1 = torch.stack(augmented_features1)  # Shape: (batch_size, height, width)
    batch_features2 = torch.stack(augmented_features2)  # Shape: (batch_size, height, width)
    
    # Generate labels indicating positive pairs (e.g., indices)
    labels = torch.arange(len(batch), device=batch_features1.device)
    
    return (batch_features1, batch_features2), labels


class SSLBrickDataset(Dataset):
    def __init__(
        self,
        zip_path_list: List[str],
        feat_path: List[str],
        sample_indicies: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        filename_list = []
        for zip_path in zip_path_list:
            zipf = ZipFile(zip_path, 'r')
            filename_list.extend(list(zipf.namelist()[1:]))

        self.filename_list = filename_list


        # Open lmdb for spectrogram loading
        self.env = lmdb.open(feat_path, readonly=True, lock=False, readahead=False)
        
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, index):
        filename = self.filename_list[index]

        with self.env.begin(write=False) as txn:
            key = filename.encode('utf-8')
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Filename '{filename}' not found in the database.")
            sxx = pickle.loads(value)
        
        # Convert to tensor
        sxx_tensor = torch.tensor(sxx, dtype=torch.float32)
        
        nan_mask = torch.isnan(sxx_tensor)
        if nan_mask.any():
            torch.where(nan_mask, torch.zeros_like(sxx_tensor), sxx_tensor)
            log.warning(f"{filename} spectrogram got nan value, filling with zero")

        # 0-1 normalization
        f_min, f_max = torch.min(sxx_tensor), torch.max(sxx_tensor)
        sxx_tensor = (sxx_tensor - f_min) / (f_max - f_min + 1e-5)

        return sxx_tensor