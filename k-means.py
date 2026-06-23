from clustering_methods import *

# ELBOW METHOD
inertia_scores = []
k_choices = range(1, 11)

for k in k_choices:
    km = KMeans(n_clusters=k).fit(X_pca)
    inertia_scores.append(km.inertia_)

kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
auto_k_val = kl.elbow

#Gap Statistic
gap_scores = []
B = 10  # Number of random datasets to generate per k

X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores[idx]
    
    # Generate and test fake, random datasets
    ref_inertias = []
    for b in range(B):
        # Create fake dots filling the exact same space
        X_ref = np.random.default_rng(seed=k+b).uniform(X_min, X_max, X_pca.shape)
        
        # Test how K-Means clusters the fake noise
        km_ref = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_ref)
        ref_inertias.append(km_ref.inertia_)
        
    # Gap = Average Fake Error - Real Error (on a log scale)
    gap = np.mean(np.log(ref_inertias)) - np.log(real_inertia)
    gap_scores.append(gap)

# Find the k value where the gap deviation is at its absolute peak
gap_k_val = k_choices[np.argmax(gap_scores)]

#Sillhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop
best_k = 2
best_score = -1
best_dbi_k = 2
best_dbi_score = float('inf')  
best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    #Silhouette Score
    temp_model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_pca)
    labels = temp_model.labels_
    sil_score = silhouette_score(X_pca, labels)
    print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
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

print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")

# K-Means Clustering
kmeans_E = KMeans(n_clusters=auto_k_val, n_init=10, random_state=42).fit(X_pca)
kmeans_S = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_pca)
kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=10, random_state=42).fit(X_pca)
kmeans_C = KMeans(n_clusters=best_ch_k, n_init=10, random_state=42).fit(X_pca)
kmeans_G = KMeans(n_clusters=gap_k_val, n_init=10, random_state=42).fit(X_pca)

models = [kmeans_E.labels_, kmeans_S.labels_, kmeans_D.labels_, kmeans_C.labels_, kmeans_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']

# Split into two windows for better comparison visualization
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 6))
for i in range(3):
    ax = axes1[i]
    current_labels = models[i]
    name = names[i]
    
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
        breakdown_parts = [f"{count} {flower_names[s_idx]}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
    
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)
fig1.tight_layout()

# --- WINDOW 2: Calinski-Harabasz, Gap Statistic) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 6))
for i in range(2):
    ax = axes2[i]
    idx = i + 3  # Offset index to grab models 3 and 4
    current_labels = models[idx]
    name = names[idx]
    
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
        breakdown_parts = [f"{count} {flower_names[s_idx]}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
    
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)
    fig2.tight_layout()

# Render both windows simultaneously
plt.show()
