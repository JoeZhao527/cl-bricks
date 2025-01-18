import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
data = torch.concat(torch.load("./prediction.pt"))

# Step 1: Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data.numpy())

# KMeans++ clustering (KMeans++ is the default initialization method in sklearn's KMeans)
# 6 clusters for 6 super parent classes
kmeans = KMeans(init='k-means++', n_clusters=6, n_jobs=12, verbose=1)  
kmeans.fit(data_normalized)

# Get the cluster labels for each sample
kmeans_labels = kmeans.labels_

# Get the cluster centers
cluster_centers = kmeans.cluster_centers_

# Compute the distance from each sample to each cluster center
distances = np.linalg.norm(data_normalized[:, np.newaxis] - cluster_centers, axis=2)

# Convert distances to a tensor
distances_tensor = torch.tensor(distances)

# Print the shape of the distances tensor
print(distances_tensor.shape)  # This should have shape (num_samples, num_clusters)

# Save the tensor
torch.save(distances_tensor, "./distances_to_centers.pt")
