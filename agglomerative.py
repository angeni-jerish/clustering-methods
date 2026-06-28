from clustering_methods import *

# 1. Define your range of distance thresholds to test
# Adjust the start, stop, and num steps based on your PCA data variance
thresholds = np.linspace(1.0, 15.0, num=30)

best_thresh = thresholds[0]
best_score = -1
best_dbi_thresh = thresholds[0]
best_dbi_score = float('inf')  # Lower is better
best_ch_thresh = thresholds[0]
best_ch_score = -1             # Higher is better

for t in thresholds:
    # Set n_clusters=None to use distance_threshold
    temp_model = AgglomerativeClustering(n_clusters=None, distance_threshold=t).fit(X_pca)
    labels = temp_model.labels_
    
    # Validation: Metrics require between 2 and (N-1) clusters to calculate properly
    num_clusters = len(np.unique(labels))
    if num_clusters < 2 or num_clusters >= len(X_pca):
        print(f"Testing threshold={t:.2f} -> Skipped (Created {num_clusters} clusters)")
        continue

    # Silhouette Score
    sil_score = silhouette_score(X_pca, labels)
    print(f"Testing threshold={t:.2f} ({num_clusters} clusters) -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_score:
        best_score = sil_score
        best_thresh = t

    # DBI
    dbi_score = davies_bouldin_score(X_pca, labels)
    print(f"Testing threshold={t:.2f} ({num_clusters} clusters) -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score:
        best_dbi_score = dbi_score
        best_dbi_thresh = t

    # CH
    ch_score = calinski_harabasz_score(X_pca, labels)
    print(f"Testing threshold={t:.2f} ({num_clusters} clusters) -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score:
        best_ch_score = ch_score
        best_ch_thresh = t

print(f"\nThe optimal threshold (By Silhouette Coefficient testing): {best_thresh:.2f}")
print(f"The optimal threshold (By Davies-Bouldin Index testing): {best_dbi_thresh:.2f}")
print(f"The optimal threshold (By Calinski-Harabasz Index testing): {best_ch_thresh:.2f}")

# Refit the models using the discovered optimal thresholds
agglo_S = AgglomerativeClustering(n_clusters=None, distance_threshold=best_thresh).fit(X_pca)
agglo_D = AgglomerativeClustering(n_clusters=None, distance_threshold=best_dbi_thresh).fit(X_pca)
agglo_C = AgglomerativeClustering(n_clusters=None, distance_threshold=best_ch_thresh).fit(X_pca)

models = [agglo_S.labels_, agglo_D.labels_, agglo_C.labels_]
names = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for i, (model, name) in enumerate(zip(models, names)):
    ax = axes[i]
    current_labels = model
    
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
    
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

plt.tight_layout()
plt.show()
