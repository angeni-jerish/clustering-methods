from clustering_methods import *
import kDBCV as dbcv
min_samples_val = 3

# K-NEAREST CLUSTERING 
# Calculate the distance to the n-th nearest neighbor for each point
neighbors = NearestNeighbors(n_neighbors=min_samples_val)
neighbors_fit = neighbors.fit(X_moons)
distances, indices = neighbors_fit.kneighbors(X_moons)
# Sorts the distances to the furthest neighbor (the last column)
k_distances = np.sort(distances[:, min_samples_val - 1])
# find the "Elbow/Knee" point on this curve
k_choices = range(len(k_distances))
kl_dbscan = KneeLocator(k_choices, k_distances, curve='convex', direction='increasing')
auto_eps = k_distances[kl_dbscan.knee]
print(f"The optimal epsilon(radius) value is (By K-Nearest Clustering): {auto_eps:.3f}")
dbscan_KN = DBSCAN(eps=auto_eps, min_samples=min_samples_val).fit(X_moons)
db_clusters = len(np.unique(dbscan_KN.labels_[dbscan_KN.labels_ != -1]))
print(f"DBSCAN found {db_clusters} clusters (excluding noise).")

eps_values = np.linspace(0.1, 0.3, 41)  # Search space for epsilon

#DBCV (Density-Based Clustering Validation) Score
best_dbcv = -1.05  # DBCV scores range from -1 to +1 (higher is better)
best_eps_dbcv = None
successful_runs = 0
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples_val).fit(X_moons)
    labels = db.labels_
    
    unique_clusters = len(np.unique(labels[labels != -1]))
    
    # DBCV requires at least 2 clusters to compute validation score
    if unique_clusters >= 2:
        try:
            # DBCV natively handles noise (-1 labels) in its mathematical graph structure
            current_dbcv, _ = dbcv.DBCV_score(X_moons, labels)
            successful_runs += 1
            # Track best parameter (+1 is perfect clustering)
            if current_dbcv > best_dbcv:
                best_dbcv = current_dbcv
                best_eps_dbcv = eps
        except Exception:
            continue

print(f"DBCV Optimized Optimal Eps: {best_eps_dbcv:.3f} (Score: {best_dbcv:.2f})")
print(f"DBCV successful runs: {successful_runs} out of {len(eps_values)} tested eps values.")
best_dbi = float('inf')
best_ch = -1
best_eps_dbi = None
best_eps_ch = None

# 2. LOOP ONLY THROUGH EPSILON
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples_val).fit(X_moons)
    labels = db.labels_
    
    # EXCLUDE NOISE (-1) FOR THE MATHEMATICAL EVALUATION
    mask_no_noise = (labels != -1)
    X_clean = X_moons[mask_no_noise]
    labels_clean = labels[mask_no_noise]
    unique_clean = len(set(labels_clean)) 
    
    # Must have at least 2 clusters to compute DBI and CH
    if unique_clean >= 2 and len(X_clean) > 0:
        current_dbi = davies_bouldin_score(X_clean, labels_clean)
        current_ch = calinski_harabasz_score(X_clean, labels_clean)
        
        # Track best parameters
        if current_dbi < best_dbi:
            best_dbi = current_dbi
            best_eps_dbi = eps
            
        if current_ch > best_ch:
            best_ch = current_ch
            best_eps_ch = eps

print(f"DBI Eps: {best_eps_dbi:.3f}")
print(f"CH Eps: {best_eps_ch:.3f}")
print(f"DBCV Eps: {best_eps_dbcv:.3f}")

dbscan_dbi = DBSCAN(eps=best_eps_dbi, min_samples=min_samples_val).fit(X_moons)
dbscan_ch = DBSCAN(eps=best_eps_ch, min_samples=min_samples_val).fit(X_moons)
dbscan_DBCV = DBSCAN(eps=best_eps_dbcv, min_samples=min_samples_val).fit(X_moons)

models = [dbscan_KN.labels_, dbscan_dbi.labels_, dbscan_ch.labels_, dbscan_DBCV.labels_]
names = ['K-Nearest Clustering', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'DBCV']

# Create the plot for your moons
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

for i, (model, name) in enumerate(zip(models, names)):
    ax = axes[i]
    current_labels = model
    unique_clusters = np.unique(current_labels)

    # Plot the 2D moons data directly using your imported X_moons matrix
    scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=current_labels, cmap='turbo', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    legend_labels = []

    # Generate custom labels mapping back to your moon ground truths
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
            
        total_pts = np.sum(current_labels == cluster_id)
        true_moons_inside = y_moons[current_labels == cluster_id]
        unique_classes, class_counts = np.unique(true_moons_inside, return_counts=True)
        
        breakdown_parts = [f"{count} {moon_names[c_idx]}" for c_idx, count in zip(unique_classes, class_counts)]
        moon_breakdown = " + ".join(breakdown_parts)
        legend_labels.append(f"Cluster {cluster_id} ({total_pts} pts: {moon_breakdown})")

    # 1. Capture and print DBSCAN outliers / noise explicitly 
    noise_count = np.sum(current_labels == -1)
    if noise_count > 0:
        noise_moons, noise_counts = np.unique(y_moons[current_labels == -1], return_counts=True)
        noise_breakdown = ", ".join([f"{c} {moon_names[s]}" for s, c in zip(noise_moons, noise_counts)])
        noise_text = f"Outliers (Noise): {noise_count} pts ({noise_breakdown})"
    else:
        noise_text = "Outliers (Noise): 0 pts"

    # 2. Extract handles from matplotlib elements
    handles, _ = scatter.legend_elements()
    
    # 3. Create an empty blank handle so the dash row has no color icon next to it
    blank_handle = plt.Line2D([], [], marker='none', linestyle='none', visible=False)

    if -1 in unique_clusters:
        # Index 0 is the outlier color handle. Index 1 onward are the valid clusters.
        final_handles = [handles[0], blank_handle] + list(handles[1:])
        final_labels = [noise_text, "—" * 32] + legend_labels
    else:
        # If no outliers exist, we use a proxy blank handle for the outlier description line
        final_handles = [blank_handle, blank_handle] + list(handles)
        final_labels = [noise_text, "—" * 32] + legend_labels

    # 4. Format the final aligned legend block position cleanly below the plot
    ax.legend(handles=final_handles, labels=final_labels, title="DBSCAN Clusters Breakdown", 
              title_fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7)

# Fixed outside the loop scope to prevent visualization panel squishing
plt.tight_layout()
plt.show()
