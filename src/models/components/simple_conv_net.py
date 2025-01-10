import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

# ------------------------
# Model Definition
# ------------------------

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=94):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: (16, 128, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces dimensions by half
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # Output: (32, 64, 32)
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 32 * 16, 128)  # Adjust based on the output size after conv layers
        self.fc2 = nn.Linear(128, n_classes)
        # Dropout (optional, helps prevent overfitting)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        # Input x: (batch_size, 1, 128, 64)
        x = x.unsqueeze(dim=1)
        
        x = self.conv1(x)  # (batch_size, 16, 128, 64)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 16, 64, 32)
        
        x = self.conv2(x)  # (batch_size, 32, 64, 32)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 32, 32, 16)
        
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 32*32*16)
        x = self.fc1(x)  # (batch_size, 128)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, n_classes)
        
        return x