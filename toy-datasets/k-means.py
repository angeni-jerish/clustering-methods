from clustering_methods import *
"""
#IRIS DATASET
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
se_scores = []  # Array to store standard errors for each k
B = 100  # Number of random datasets to generate per k
X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores[idx]
    ref_log_inertias = []
    
    # Generate reproducible uniform noise
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min, X_max, X_pca.shape)
        km_ref = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_ref)
        ref_log_inertias.append(np.log(km_ref.inertia_))    
    
    # 1. Calculate Gap(k)
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores.append(gap)
    
    # 2. Calculate Standard Error SE(k) 
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores.append(se)

# 3. Apply the 1-SE Rule Condition sequentially from k=1 onwards
gap_k_val = k_choices[-1]  # Fallback to max k if condition is never met

for i in range(len(k_choices) - 1):
    k_curr = k_choices[i]
    gap_curr = gap_scores[i]
    
    gap_next = gap_scores[i + 1]
    se_next = se_scores[i + 1]
    
    # Condition: Gap(k) >= Gap(k+1) - SE(k+1)
    if gap_curr >= gap_next - se_next:
        gap_k_val = k_curr
        break

#Sillhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop
best_k = 2
best_score = -1
best_dbi_k = 2
best_dbi_score = float('inf')  
best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    #Silhouette Score
    temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_pca)
    labels = temp_model.labels_
    sil_score = silhouette_score(X_pca, labels)
    print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_score:
        best_score = sil_score
        best_k = k

    #DBI
    dbi_score = davies_bouldin_score(X_pca, labels)
    print(f"Testing k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, best_dbi_k = dbi_score, k

    #CH
    ch_score = calinski_harabasz_score(X_pca, labels)
    print(f"Testing k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, best_ch_k = ch_score, k

print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")

# K-Means Clustering Models
kmeans_E = KMeans(n_clusters=auto_k_val, n_init=50, random_state=42).fit(X_pca)
kmeans_S = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X_pca)
kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=50, random_state=42).fit(X_pca)
kmeans_C = KMeans(n_clusters=best_ch_k, n_init=50, random_state=42).fit(X_pca)
kmeans_G = KMeans(n_clusters=gap_k_val, n_init=50, random_state=42).fit(X_pca)

models = [kmeans_E.labels_, kmeans_S.labels_, kmeans_D.labels_, kmeans_C.labels_, kmeans_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']
fig, axes = plt.subplots(1, 5, figsize=(25, 6))

# Loop through all 5 models sequentially
for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    name = names[i]
    
    # Generate the scatter plot
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    
    # Calculate the composition breakdown for the legend
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        true_species_inside = iris.target[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} {flower_names[s_idx]}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    # Apply the legend below each individual subplot
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

# Adjust layout once for the entire single figure window
fig.tight_layout()

# ==========================================
# 2. MOONS DATASET PROCESSING (FIXED)
# ==========================================
# ==========================================
# 2. MOONS DATASET PROCESSING (FIXED & COMPLETED)
# ==========================================

# ELBOW METHOD (Calculated specifically for X_moons)
inertia_scores_moons = []
k_choices = range(1, 11)
for k in k_choices:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X_moons)
    inertia_scores_moons.append(km.inertia_)
kl_moons = KneeLocator(k_choices, inertia_scores_moons, curve='convex', direction='decreasing')
moons_elbow_k = kl_moons.elbow if kl_moons.elbow is not None else 2

# Gap Statistic (Calculated specifically for X_moons)
gap_scores_moons = []
se_scores_moons = []  
B = 100  
X_min_moons = X_moons.min(axis=0)
X_max_moons = X_moons.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores_moons[idx]
    ref_log_inertias = []
    
    # Generate reproducible uniform noise
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min_moons, X_max_moons, X_moons.shape)
        km_ref = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_ref)
        ref_log_inertias.append(np.log(km_ref.inertia_))    
    
    # 1. Calculate Gap(k)
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores_moons.append(gap)
    
    # 2. Calculate Standard Error SE(k) 
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores_moons.append(se)

# 3. Apply the 1-SE Rule Condition sequentially from k=1 onwards
moons_gap_k = k_choices[-1]  

for i in range(len(k_choices) - 1):
    k_curr = k_choices[i]
    gap_curr = gap_scores_moons[i]
    
    gap_next = gap_scores_moons[i + 1]
    se_next = se_scores_moons[i + 1]
    
    if gap_curr >= gap_next - se_next:
        moons_gap_k = k_curr
        break

# Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop (Moons)
moons_best_silhouette_k = 2
best_sil_score = -1
moons_best_dbi_k = 2
best_dbi_score = float('inf')  
moons_best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    # Fit model specifically on X_moons
    temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_moons)
    labels = temp_model.labels_
    
    # Silhouette Score
    sil_score = silhouette_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_sil_score = sil_score
        moons_best_silhouette_k = k

    # DBI
    dbi_score = davies_bouldin_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, moons_best_dbi_k = dbi_score, k

    # CH
    ch_score = calinski_harabasz_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, moons_best_ch_k = ch_score, k

# K-Means Clustering Models (Moons)
kmeans_E = KMeans(n_clusters=moons_elbow_k, n_init=50, random_state=42).fit(X_moons)
kmeans_S = KMeans(n_clusters=moons_best_silhouette_k, n_init=50, random_state=42).fit(X_moons)
kmeans_D = KMeans(n_clusters=moons_best_dbi_k, n_init=50, random_state=42).fit(X_moons)
kmeans_C = KMeans(n_clusters=moons_best_ch_k, n_init=50, random_state=42).fit(X_moons)
kmeans_G = KMeans(n_clusters=moons_gap_k, n_init=50, random_state=42).fit(X_moons)

models = [kmeans_E.labels_, kmeans_S.labels_, kmeans_D.labels_, kmeans_C.labels_, kmeans_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Moons Dataset Clustering Comparisons", fontsize=16, fontweight='bold')

# Loop through all 5 models sequentially
for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    name = names[i]
    
    # Generate the scatter plot
    scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Calculate the composition breakdown for the legend (FIXED: Uses ground truth moon array)
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        # Using a generic indicator array from your ground truth moons variable (e.g., y_moons)
        true_species_inside = y_moons[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Class {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    # Apply the legend below each individual subplot
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

# Adjust layout once for the entire single figure window
fig.tight_layout()

"""
# ==========================================
# 3. BLOBS DATASET PROCESSING (FIXED & COMPLETED)
# ==========================================

# ELBOW METHOD (FIXED: Changed .fit(X_moons) to .fit(X_blobs))
inertia_scores = []
k_choices = range(1, 11)
for k in k_choices:
    km = KMeans(n_clusters=k).fit(X_blobs)
    inertia_scores.append(km.inertia_)
kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
auto_k_val = kl.elbow

# Gap Statistic
gap_scores = []
se_scores = []  
B = 100  
X_min = X_blobs.min(axis=0)
X_max = X_blobs.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores[idx]
    ref_log_inertias = []
    
    # Generate reproducible uniform noise
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min, X_max, X_blobs.shape)
        km_ref = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_ref)
        ref_log_inertias.append(np.log(km_ref.inertia_))    
    
    # 1. Calculate Gap(k)
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores.append(gap)
    
    # 2. Calculate Standard Error SE(k) 
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores.append(se)

# 3. Apply the 1-SE Rule Condition sequentially from k=1 onwards
gap_k_val = k_choices[-1]  

for i in range(len(k_choices) - 1):
    k_curr = k_choices[i]
    gap_curr = gap_scores[i]
    
    gap_next = gap_scores[i + 1]
    se_next = se_scores[i + 1]
    
    if gap_curr >= gap_next - se_next:
        gap_k_val = k_curr
        break

# Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop
best_k = 2
best_score = -1
best_dbi_k = 2
best_dbi_score = float('inf')  
best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    # Silhouette Score
    temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_blobs)
    labels = temp_model.labels_
    sil_score = silhouette_score(X_blobs, labels)
    print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_score:
        best_score = sil_score
        best_k = k

    # DBI
    dbi_score = davies_bouldin_score(X_blobs, labels)
    print(f"Testing k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, best_dbi_k = dbi_score, k

    # CH
    ch_score = calinski_harabasz_score(X_blobs, labels)
    print(f"Testing k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, best_ch_k = ch_score, k

print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")

# K-Means Clustering Models (COMPLETED)
kmeans_E = KMeans(n_clusters=auto_k_val, n_init=50, random_state=42).fit(X_blobs)
kmeans_S = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X_blobs)
kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=50, random_state=42).fit(X_blobs)
kmeans_C = KMeans(n_clusters=best_ch_k, n_init=50, random_state=42).fit(X_blobs)
kmeans_G = KMeans(n_clusters=gap_k_val, n_init=50, random_state=42).fit(X_blobs)

models_blobs = [kmeans_E.labels_, kmeans_S.labels_, kmeans_D.labels_, kmeans_C.labels_, kmeans_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']

print(f"BLOBS: Total Clusters Generated: {num_clusters}")
print(f"BLOBS: Total Data Points: {num_samples}")
print(f"BLOBS: Cluster Spreads (Densities): {[round(s, 2) for s in surprise_stds]}")
unique_blobs, blob_counts = np.unique(y_blobs, return_counts=True)
for b_id, count in zip(unique_blobs, blob_counts):
    percentage = (count / num_samples) * 100
    print(f"  Blob {b_id}: {count} points ({percentage:.1f}%)")

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Blobs Dataset Clustering Comparisons", fontsize=16, fontweight='bold')

# Loop through all 5 blob models sequentially
for i in range(5):
    ax = axes[i]
    current_labels = models_blobs[i]
    name = names[i]
    
    # Generate the scatter plot for blobs
    scatter = ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Calculate the composition breakdown for the legend (Using ground truth blob array)
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        # Using a generic indicator array from your ground truth blobs variable (e.g., y_blobs)
        true_species_inside = y_blobs[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Class {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    # Apply the legend below each individual subplot
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

fig.tight_layout()
"""
# ==========================================
# 2. CIRCLES DATASET PROCESSING (FIXED)
# ==========================================

# ELBOW METHOD (Calculated specifically for X_circles)
inertia_scores_circles = []
k_choices = range(1, 11)
for k in k_choices:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X_circles)
    inertia_scores_circles.append(km.inertia_)
kl_circles = KneeLocator(k_choices, inertia_scores_circles, curve='convex', direction='decreasing')
circles_elbow_k = kl_circles.elbow if kl_circles.elbow is not None else 2

# Gap Statistic (Calculated specifically for X_circles)
gap_scores_circles = []
se_scores_circles = []  
B = 100  
X_min_circles = X_circles.min(axis=0)
X_max_circles = X_circles.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores_circles[idx]
    ref_log_inertias = []
    
    # Generate reproducible uniform noise
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min_circles, X_max_circles, X_circles.shape)
        km_ref = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_ref)
        ref_log_inertias.append(np.log(km_ref.inertia_))    
    
    # 1. Calculate Gap(k)
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores_circles.append(gap)
    
    # 2. Calculate Standard Error SE(k) 
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores_circles.append(se)

# 3. Apply the 1-SE Rule Condition sequentially from k=1 onwards
circles_gap_k = k_choices[-1]  

for i in range(len(k_choices) - 1):
    k_curr = k_choices[i]
    gap_curr = gap_scores_circles[i]
    
    gap_next = gap_scores_circles[i + 1]
    se_next = se_scores_circles[i + 1]
    
    if gap_curr >= gap_next - se_next:
        circles_gap_k = k_curr
        break

# Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop (Circles)
circles_best_silhouette_k = 2
best_sil_score = -1
circles_best_dbi_k = 2
best_dbi_score = float('inf')  
circles_best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    # Fit model specifically on X_circles
    temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_circles)
    labels = temp_model.labels_
    
    # Silhouette Score
    sil_score = silhouette_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_sil_score = sil_score
        circles_best_silhouette_k = k

    # DBI
    dbi_score = davies_bouldin_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, circles_best_dbi_k = dbi_score, k

    # CH
    ch_score = calinski_harabasz_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, circles_best_ch_k = ch_score, k

# K-Means Clustering Models (Circles)
kmeans_E = KMeans(n_clusters=auto_k_val, n_init=50, random_state=42).fit(X_circles)
kmeans_S = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X_circles)
kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=50, random_state=42).fit(X_circles)
kmeans_C = KMeans(n_clusters=best_ch_k, n_init=50, random_state=42).fit(X_circles)
kmeans_G = KMeans(n_clusters=gap_k_val, n_init=50, random_state=42).fit(X_circles)

models = [kmeans_E.labels_, kmeans_S.labels_, kmeans_D.labels_, kmeans_C.labels_, kmeans_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Circles Dataset Clustering Comparisons", fontsize=16, fontweight='bold')

# Loop through all 5 models sequentially
for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    name = names[i]
    
    # Generate the scatter plot
    scatter = ax.scatter(X_circles[:, 0], X_circles[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{name} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Calculate the composition breakdown for the legend (FIXED: Uses ground truth circle array)
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        # Using a generic indicator array from your ground truth circles variable (e.g., y_circles)
        true_species_inside = y_circles[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Class {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    # Apply the legend below each individual subplot
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

# Adjust layout once for the entire single figure window
fig.tight_layout()
"""
plt.show()
