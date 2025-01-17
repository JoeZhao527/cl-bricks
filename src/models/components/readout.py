import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

# ------------------------
# Model Definition
# ------------------------

class MLPReadout(nn.Module):
    def __init__(self, n_classes=94, stat_dim=162):
        super(MLPReadout, self).__init__()

        # Statistical info encoder
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.readout = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )
    
    def forward(self, x):
        spec_feat, stat_feat = x

        # z = self.stat_encoder(stat_feat)

        # out = self.readout(torch.concat([x, z], dim=1))
        out = self.readout(spec_feat)

        return out