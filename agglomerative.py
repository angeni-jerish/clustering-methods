from clustering_methods import *

#SILLHOUETTE SCORE
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

models = agglo.labels_
name = 'Agglomerative'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))