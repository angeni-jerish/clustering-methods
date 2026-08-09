import random as py_random  

import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

#fixing issue with DBCV
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# Load the Iris dataset
iris = load_iris() 
X = iris.data
flower_names = ['Setosa', 'Versicolor', 'Virginica']
# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

#PCA (making 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# MOONS DATASET FOR DBSCAN
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
moon_names = ['Bottom Crescent', 'Top Crescent']

# CIRCLES DATASET FOR DBSCAN
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, random_state=42)
circle_names = ['Inner Circle', 'Outer Circle']

# BLOBS DATASET
# 1. Generate the surprise configuration
num_clusters = py_random.randint(2, 6)       # Randomly choose between 2 and 6 clusters
num_samples = py_random.randint(300, 1000)   # Randomly choose between 300 and 1000 data points

# Create a surprise array of random standard deviations (densities) for each cluster
surprise_stds = [py_random.uniform(0.3, 2.2) for _ in range(num_clusters)]

# 2. Feed the random parameters into make_blobs
X_blobs, y_blobs = make_blobs(
    n_samples=num_samples,
    centers=num_clusters,
    n_features=2,                         # Kept at 2D 
    cluster_std=surprise_stds
)
