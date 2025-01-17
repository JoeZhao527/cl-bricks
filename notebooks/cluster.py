import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

data = torch.concat(torch.load("./prediction.pt"))

# Step 1: Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.numpy())

# DBSCAN clustering (no need to specify the number of clusters)
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(data_normalized)

# Convert DBSCAN labels to a tensor
dbscan_labels_tensor = torch.tensor(dbscan_labels)

print(dbscan_labels_tensor.shape)
torch.save(dbscan_labels_tensor, "./cluster.pt")