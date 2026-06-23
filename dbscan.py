from clustering_methods import *

#K NEAREST NEIGHBOURS 
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

models = dbscan.labels_
name = 'DBSCAN'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))