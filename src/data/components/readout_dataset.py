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

def collate_fn(batch):
    # Unzip the batch into features and labels
    spec_feat, time_feat, labels = zip(*batch)
    
    # Stack all padded features into a single tensor
    batch_spec_features = torch.stack(spec_feat)
    batch_time_features = torch.stack(time_feat)
    batch_labels = torch.stack(labels)

    return (batch_spec_features, batch_time_features), batch_labels

class BrickDataset(Dataset):
    def __init__(
        self,
        zip_path: str,
        feat_path: str,
        stat_path: str,
        inference_set: bool = False,
        label_path: str = None,
        sample_indicies: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        zipf = ZipFile(zip_path, 'r')
        stat_feat = pd.read_csv(stat_path)

        if label_path != None:
            label_df = pd.read_csv(label_path)
            label = {"train_X/" + rec[0]: rec[1:] for rec in label_df.values}


        filename_list = list(zipf.namelist()[1:])
        self.filename_list = filename_list

        stat_feat = stat_feat.values
        self.stat_feat = {
            f_name: torch.tensor(stat_feat[i], dtype=torch.float32)
            for i, f_name in enumerate(filename_list)
        }

        all_feat = torch.concat(torch.load(feat_path), dim=0)
        all_feat = all_feat[len(label_df):] if inference_set else all_feat[:len(label_df)]

        self.spec_feat = {
            f_name: all_feat[i]
            for i, f_name in enumerate(filename_list)
        }

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
        spec_feat = self.spec_feat[filename]

        return spec_feat, timespace_feat, label