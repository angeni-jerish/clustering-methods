import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the Iris dataset
iris = load_iris() 
X = iris.data
# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

#PCA (making 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Using Elbow method for k-means
inertia_scores = []
k_choices = range(1, 11)

for k in k_choices:
    km = KMeans(n_clusters=k, random_state=30).fit(X_pca)
    inertia_scores.append(km.inertia_)

kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
auto_k_val = kl.elbow

print(f"The optimal num of clusters (By Elbow method, k-means clustering): {auto_k_val}")

# K-Means Clustering
kmeans = KMeans(n_clusters=auto_k_val, random_state=30).fit(X_pca)

#using Silhouette Score for Agglomerative
best_k = 2
best_score = -1

for k in range(2, 11):
    # Temporarily fit a model with k clusters
    temp_model = AgglomerativeClustering(n_clusters=k).fit(X_pca)
    labels = temp_model.labels_
    
    # Calculate the silhouette score for this configuration
    score = silhouette_score(X_pca, labels)
    print(f"Testing k={k} -> Silhouette Score: {score:.3f}")
    
    # Track the highest score
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nThe optimal num of clusters (By Silhouette Coefficient testing, Agglomerative clustering): {best_k}")

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=best_k).fit(X_pca)

#K Nearest Neighbours for DBSCAN
min_samples_val = 5

# Calculate the distance to the n-th nearest neighbor for each point
neighbors = NearestNeighbors(n_neighbors=min_samples_val)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)

# SOrts the distances to the furthest neighbor (the last column)
k_distances = np.sort(distances[:, min_samples_val - 1])

# find the "Elbow/Knee" point on this curve
k_choices = range(len(k_distances))
kl_dbscan = KneeLocator(k_choices, k_distances, curve='convex', direction='increasing')

auto_eps = k_distances[kl_dbscan.knee]
print(f"The optimal epsilon(radius) value is (By K-Nearest Neighbours, DBSAN Clustering): {auto_eps:.3f}")

dbscan = DBSCAN(eps=auto_eps, min_samples=min_samples_val).fit(X_pca)

db_clusters = len(np.unique(dbscan.labels_[dbscan.labels_ != -1]))
print(f"DBSCAN found {db_clusters} clusters (excluding noise).")

# Plotting the graph using the new PC1 and PC2 labels
models = [kmeans.labels_, agglo.labels_, dbscan.labels_]
names = ['K-Means', 'Agglomerative', 'DBSCAN']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=models[i], cmap='viridis', edgecolors='k')
    ax.set_title(f"{names[i]}")
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")

plt.tight_layout()
plt.show()



