import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np

# Load the Iris dataset
iris = load_iris() 
X = iris.data
# Standardize the features
X_scaled = StandardScaler().fit_transform(X)
# K-Means Clustering
k_val=len(np.unique(iris.target)) 
kmeans = KMeans(n_clusters=k_val, random_state=30).fit(X_scaled)
# Agglomerative Clustering
agglo = AgglomerativeClustering(distance_threshold=10, n_clusters=None).fit(X_scaled)

# agglo clusters found:
print(f"Number of clusters found: {agglo.n_clusters_}")
# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)

# 3. Use NumPy to analyze the DBSCAN discovery
db_clusters = len(np.unique(dbscan.labels_[dbscan.labels_ != -1]))
print(f"DBSCAN found {db_clusters} clusters (excluding noise).")


# We'll plot Sepal Length (index 0) vs Sepal Width (index 1)
models = [kmeans.labels_, agglo.labels_, dbscan.labels_]
names = ['K-Means', 'Agglomerative', 'DBSCAN']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    ax.scatter(X[:, 0], X[:, 1], c=models[i], cmap='viridis', edgecolors='k')
    ax.set_title(f"{names[i]}")
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()



