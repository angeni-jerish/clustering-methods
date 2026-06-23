from clustering_methods import *

#K NEAREST CLUSTERING 
min_samples_val = 5
# Calculate the distance to the n-th nearest neighbor for each point
neighbors = NearestNeighbors(n_neighbors=min_samples_val)
neighbors_fit = neighbors.fit(X_pca)
distances, indices = neighbors_fit.kneighbors(X_pca)
# Sorts the distances to the furthest neighbor (the last column)
k_distances = np.sort(distances[:, min_samples_val - 1])
# find the "Elbow/Knee" point on this curve
k_choices = range(len(k_distances))
kl_dbscan = KneeLocator(k_choices, k_distances, curve='convex', direction='increasing')
auto_eps = k_distances[kl_dbscan.knee]
print(f"The optimal epsilon(radius) value is (By K-Nearest Clustering): {auto_eps:.3f}")
dbscan = DBSCAN(eps=auto_eps, min_samples=min_samples_val).fit(X_pca)
db_clusters = len(np.unique(dbscan.labels_[dbscan.labels_ != -1]))
print(f"DBSCAN found {db_clusters} clusters (excluding noise).")


models = [dbscan.labels_]
names = ['K-Nearest Clustering']
fig, ax = plt.subplots(figsize=(8, 6))

for i, (models, name) in enumerate(zip(models, names)):
    current_labels = models
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    unique_clusters = np.unique(current_labels)
    legend_labels = []
    
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        true_species_inside = iris.target[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        
        breakdown_parts = []
        for s_idx, count in zip(unique_species, species_counts):
            breakdown_parts.append(f"{count} {flower_names[s_idx]}")
            
        species_breakdown = " + ".join(breakdown_parts)
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {species_breakdown})")
    
    # Isolate and explicitly show DBSCAN noise as an external textbox
    if names[i] == 'DBSCAN':
        is_noise = (current_labels == -1)
        noise_count = np.sum(is_noise)
        
        if noise_count > 0:
            noise_species, noise_counts = np.unique(iris.target[is_noise], return_counts=True)
            noise_breakdown = ", ".join([f"{c} {flower_names[s]}" for s, c in zip(noise_species, noise_counts)])
            
            ax.text(0.50,0.05,f"Outliers (Noise): ({noise_count}\n({noise_breakdown})", 
                    transform=ax.transAxes, fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
    handles, _ = scatter.legend_elements()
    # Synchronize the handles to filter out the noise marker if it exists in legend handles
    if -1 in unique_clusters:
        # Drop the first color handle (the purple noise handle) to align the rest
        ax.legend(handles=handles[1:], labels=legend_labels, title="Model Clusters Breakdown", loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7)
    else:
        ax.legend(handles=handles, labels=legend_labels, title="Model Clusters Breakdown", loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7)

plt.tight_layout()
plt.show()