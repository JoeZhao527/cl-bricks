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


def collate_wo_augmentation(batch: List[torch.Tensor], target_dim: Tuple[int, int] = (128, 64)):
    """
    Collate function to generate a batch suitable for whole data feature extraction
    """
    # target_dim max is (129, 71)
    padded_features = []
    target_height, target_width = target_dim  # Target dimensions

    for idx, feat in enumerate(batch):
        # Ensure the feature is a torch.Tensor
        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"Feature at index {idx} is not a torch.Tensor. Got {type(feat)}")

        # Get current dimensions
        current_height, current_width = feat.shape
        
        # Calculate padding sizes
        pad_height = target_height - current_height
        pad_width = target_width - current_width
        
        # Initialize padding for height and width
        # pad should be in the format (pad_left, pad_right, pad_top, pad_bottom)
        pad = (0, pad_width, 0, pad_height)
        
        # If padding is needed
        if pad_height > 0 or pad_width > 0:
            # Apply padding with constant value 0
            padded_feat = F.pad(feat, pad, mode='constant', value=0)
        else:
            # If the feature is larger than target, truncate it
            padded_feat = feat[:target_height, :target_width]
        
        # Ensure the padded feature has the target shape
        assert padded_feat.shape == (target_height, target_width), \
            f"Padded feature has shape {padded_feat.shape}, expected {(target_height, target_width)}"
        
        padded_features.append(padded_feat)
    
    # Stack all padded features into a single tensor
    batch_spec_features = torch.stack(padded_features)

    return batch_spec_features


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
        
        # 0-1 normalization
        f_min, f_max = np.min(sxx), np.max(sxx)
        sxx = (sxx - f_min) / (f_max - f_min + 1e-5)

        # Convert to tensor
        sxx_tensor = torch.tensor(sxx, dtype=torch.float32)

        nan_mask = torch.isnan(sxx_tensor)
        if nan_mask.any():
            sxx_tensor = torch.where(nan_mask, torch.zeros_like(sxx_tensor), sxx_tensor)
            print(f"{filename} spectrogram got nan value, filling with zero")

        return sxx_tensor