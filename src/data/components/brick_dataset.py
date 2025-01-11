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
from typing import Tuple
import lmdb
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

# from src.data.components.featurizer import timestamp_feature_extraction, signal_stat_feature_extraction

# def get_timespace_feature(datapoint) -> Tuple[str, torch.Tensor]:
#     timestamp = datapoint['t'].astype('timedelta64[s]').astype(int)
#     timestamp_feat = timestamp_feature_extraction(timestamp)

#     signal = datapoint['v']
#     signal_feat = signal_stat_feature_extraction(signal)

#     timespace_feat = np.concatenate([
#         np.array(list(timestamp_feat.values())),
#         np.array(list(signal_feat.values())),
#     ])
    
#     return torch.Tensor(timespace_feat)

def collate_fn(batch, target_dim: Tuple[int, int] = (128, 64)):
    # target_dim max is (129, 71)
    
    # Unzip the batch into features and labels
    # spec_feat: spectrogram, dimension (time, frequency)
    # time_feat: statistical values, 28-D tensor
    spec_feat, time_feat, labels = zip(*batch)
    
    padded_features = []
    target_height, target_width = target_dim  # Target dimensions
    
    for idx, feat in enumerate(spec_feat):
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
    batch_time_features = torch.stack(time_feat)

    batch_labels = torch.stack(labels)

    return (batch_spec_features, batch_time_features), batch_labels

class BrickDataset(Dataset):
    def __init__(
        self,
        zip_path: str,
        feat_path: str,
        stat_path: str,
        label_path: str = None,
        sample_indicies: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        zipf = ZipFile(zip_path, 'r')
        stat_feat = pd.read_csv(stat_path)

        # if the dataset has label path (provided training set), get filenames from label file
        if label_path != None:
            label_df = pd.read_csv(label_path)
            filename_list = label_df['filename'].apply(lambda x: "train_X/" + x)
            label = {"train_X/" + rec[0]: rec[1:] for rec in label_df.values}

        # if the dataset does not has label path (provided test set), get filenames from the zip
        else:
            filename_list = zipf.namelist()[1:]

        filename_list = pd.Series(filename_list)
        # if sample indicies provided, only use a part of the dataset
        # Otherwise use the full dataset
        if sample_indicies:
            filename_list = filename_list.iloc[[sample_indicies]]
            stat_feat = stat_feat.iloc[[sample_indicies]]

        filename_list = list(filename_list)
        self.filename_list = filename_list

        stat_feat = stat_feat.values
        self.stat_feat = {
            f_name: torch.tensor(stat_feat[i], dtype=torch.float32)
            for i, f_name in enumerate(filename_list)
        }

        # Open lmdb for spectrogram loading
        self.env = lmdb.open(feat_path, readonly=True, lock=False, readahead=False)

        # Prepare labels
        if label_path != None:
            self.label = {
                f_name: (torch.tensor(label[f_name].astype(int)) >= 0).float()
                for f_name in filename_list
            }
        else:
            self.label = {}
        
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, index):
        filename = self.filename_list[index]
        label = self.label.get(filename, torch.Tensor([0]))
        timespace_feat = self.stat_feat[filename]

        with self.env.begin(write=False) as txn:
            key = filename.encode('utf-8')
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Filename '{filename}' not found in the database.")
            sxx = pickle.loads(value)
        
        # Convert to tensor
        sxx_tensor = torch.tensor(sxx, dtype=torch.float32)
        
        # 0-1 normalization
        f_min, f_max = torch.min(sxx_tensor), torch.max(sxx_tensor)
        sxx_tensor = (sxx_tensor - f_min) / (f_max - f_min + 1e-5)

        return sxx_tensor, timespace_feat, label