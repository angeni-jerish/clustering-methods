from clustering_methods import *

best_k = 2
best_score = -1
best_dbi_k = 2
best_dbi_score = float('inf')  # Start at infinity because lower is better
best_ch_k = 2
best_ch_score = -1             # Start at -1 because higher is better

for k in range(2, 11):
    #Silhouette Score
    # Temporarily fit a model with k clusters
    temp_model = AgglomerativeClustering(n_clusters=k).fit(X_pca)
    labels = temp_model.labels_
    # Calculate the silhouette score for this configuration
    sil_score = silhouette_score(X_pca, labels)
    print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
    # Track the highest score
    if sil_score > best_score:
        best_score = sil_score
        best_k = k

    #DBI
    dbi_score = davies_bouldin_score(X_pca, labels)
    print(f"Testing k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score:  # Lower is better!
        best_dbi_score, best_dbi_k = dbi_score, k

    #CH
    ch_score = calinski_harabasz_score(X_pca, labels)
    print(f"Testing k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score:  # Higher is better!
        best_ch_score, best_ch_k = ch_score, k

print(f"\nThe optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")

# Agglomerative Clustering
agglo_S = AgglomerativeClustering(n_clusters=best_k).fit(X_pca)
agglo_D = AgglomerativeClustering(n_clusters=best_dbi_k).fit(X_pca)
agglo_C = AgglomerativeClustering(n_clusters=best_ch_k).fit(X_pca)


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
    
    # Place a clean legend below each individual subplot
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

plt.tight_layout()
plt.show()