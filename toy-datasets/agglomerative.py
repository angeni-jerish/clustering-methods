import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator
from clustering_methods import *

# --- Calculates WCSS (inertia) using the pairwise-distance identity. ---
def get_agglomerative_inertia(X, labels):
    total_inertia = 0
    unique_labels = np.unique(labels)
    
    # Precompute squared distances to maximize looping performance
    dist_matrix = pairwise_distances(X, metric='euclidean') ** 2
    
    for label in unique_labels:
        # Isolate indices for points belonging to this cluster
        indices = np.where(labels == label)[0]
        n_k = len(indices)
        
        if n_k > 1:
            # Extract cluster distance sub-matrix
            cluster_distances = dist_matrix[indices][:, indices]
            # Pairwise identity calculation: sum / (2 * n_points)
            total_inertia += np.sum(cluster_distances) / (2 * n_k)
            
    return total_inertia

# --- Configuration Setup ---
thresholds = np.linspace(1.0, 15.0, num=30)
chosen_linkage = 'ward' 
"""
# --- ELBOW METHOD ---
inertia_scores = []
valid_elbow_thresholds = []

for t in thresholds:
    km = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage=chosen_linkage).fit(X_pca)
    num_clusters = len(np.unique(km.labels_))
    
    # Inertia is mathematically meaningless for 1 cluster or N singletons
    if 1 < num_clusters < len(X_pca):
        inertia = get_agglomerative_inertia(X_pca, km.labels_)
        inertia_scores.append(inertia)
        valid_elbow_thresholds.append(t)

# Run KneeLocator only on the thresholds that yielded calculable inertia
kl = KneeLocator(valid_elbow_thresholds, inertia_scores, curve='convex', direction='decreasing')
best_e_thresh = kl.elbow if kl.elbow is not None else thresholds[0]

# --- REVISED GAP STATISTIC (WITH 1-SE RULE) ---
gap_scores_dict = {}
se_scores_dict = {}
k_to_threshold = {}

B = 100  # Number of reference datasets
X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)

for t in thresholds:
    # 1. Real Data Inertia
    km_real = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage=chosen_linkage).fit(X_pca)
    k = len(np.unique(km_real.labels_))
    
    if k <= 1 or k >= len(X_pca):
        continue
        
    # Keep track of the first/best threshold that generates this specific 'k'
    if k not in k_to_threshold:
        k_to_threshold[k] = t
        
    real_inertia = get_agglomerative_inertia(X_pca, km_real.labels_)
    
    # 2. Reference Noise Inertia Collection (Forced to the SAME 'k' clusters)
    ref_log_inertias = []
    rng = np.random.default_rng(seed=int(t * 100)) 
    
    for b in range(B):
        X_ref = rng.uniform(X_min, X_max, X_pca.shape)
        # CRITICAL FIX: Force the reference data to have exactly 'k' clusters, not threshold 't'
        km_ref = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_ref)
        
        ref_inertia = get_agglomerative_inertia(X_ref, km_ref.labels_)
        ref_log_inertias.append(np.log(ref_inertia))
            
    # Calculate gap score and standard error for this specific cluster count 'k'
    if len(ref_log_inertias) > 0:
        gap = np.mean(ref_log_inertias) - np.log(real_inertia)
        
        # Standard Error formula: SDK * sqrt(1 + 1/B)
        sdk = np.std(ref_log_inertias, ddof=1)
        se = sdk * np.sqrt(1 + 1.0 / B)
        
        gap_scores_dict[k] = gap
        se_scores_dict[k] = se

# 3. Apply the 1-SE Rule Condition sequentially
sorted_k = sorted(gap_scores_dict.keys())
best_k = sorted_k[-1]  # Fallback to the maximum k found

for idx, k in enumerate(sorted_k[:-1]):
    next_k = sorted_k[idx + 1]
    
    # The condition from your image: Gap(k) >= Gap(k+1) - SE(k+1)
    if gap_scores_dict[k] >= gap_scores_dict[next_k] - se_scores_dict[next_k]:
        best_k = k
        break

# Map the winning 'k' back to your optimal threshold value
best_gap_thresh = k_to_threshold[best_k]

# (SILHOUETTE, DBI, CH) 
best_thresh = thresholds[0]
best_score = -1
best_dbi_thresh = thresholds[0]
best_dbi_score = float('inf')
best_ch_thresh = thresholds[0]
best_ch_score = -1

for t in thresholds:
    temp_model = AgglomerativeClustering(n_clusters=None, distance_threshold=t, linkage=chosen_linkage).fit(X_pca)
    labels = temp_model.labels_
    num_clusters = len(np.unique(labels))
    
    if num_clusters < 2 or num_clusters >= len(X_pca):
        print(f"Testing threshold={t:.2f} -> Skipped (Created {num_clusters} clusters)")
        continue

    # Silhouette Score
    sil_score = silhouette_score(X_pca, labels)
    if sil_score > best_score:
        best_score = sil_score
        best_thresh = t

    # DBI
    dbi_score = davies_bouldin_score(X_pca, labels)
    if dbi_score < best_dbi_score:
        best_dbi_score = dbi_score
        best_dbi_thresh = t

    # CH Index
    ch_score = calinski_harabasz_score(X_pca, labels)
    if ch_score > best_ch_score:
        best_ch_score = ch_score
        best_ch_thresh = t

# --- OUTPUT AND REFITTING ---
print(f"\nThe optimal threshold (By Elbow method): {best_e_thresh:.2f}")
print(f"The optimal threshold (By Gap Statistic method): {best_gap_thresh:.2f}")
print(f"The optimal threshold (By Silhouette Coefficient testing): {best_thresh:.2f}")
print(f"The optimal threshold (By Davies-Bouldin Index testing): {best_dbi_thresh:.2f}")
print(f"The optimal threshold (By Calinski-Harabasz Index testing): {best_ch_thresh:.2f}")

# Re-fit models using verified thresholds
agglo_E = AgglomerativeClustering(n_clusters=None, distance_threshold=best_e_thresh, linkage=chosen_linkage).fit(X_pca)
agglo_G = AgglomerativeClustering(n_clusters=None, distance_threshold=best_gap_thresh, linkage=chosen_linkage).fit(X_pca)
agglo_S = AgglomerativeClustering(n_clusters=None, distance_threshold=best_thresh, linkage=chosen_linkage).fit(X_pca)
agglo_D = AgglomerativeClustering(n_clusters=None, distance_threshold=best_dbi_thresh, linkage=chosen_linkage).fit(X_pca)
agglo_C = AgglomerativeClustering(n_clusters=None, distance_threshold=best_ch_thresh, linkage=chosen_linkage).fit(X_pca)

# --- VISUALIZATION LOOP ---
models = [agglo_E.labels_, agglo_G.labels_, agglo_S.labels_, agglo_D.labels_, agglo_C.labels_]
names = ['Elbow Method', 'Gap Statistic', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
fig, axes = plt.subplots(1, 5, figsize=(24, 6))

for i, (model_labels, name) in enumerate(zip(models, names)):
    ax = axes[i]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=model_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    unique_clusters = np.unique(model_labels)
    legend_labels = []
    
    for cluster_id in unique_clusters:
        total_pts = np.sum(model_labels == cluster_id)
        true_species_inside = iris.target[model_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        
        breakdown_parts = [f"{count} {flower_names[s_idx]}" for s_idx, count in zip(unique_species, species_counts)]
        species_breakdown = " + ".join(breakdown_parts)
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {species_breakdown})")
    
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)

plt.tight_layout()

# ==========================================
# 2. MOONS DATASET PROCESSING (AGGLOMERATIVE)
# ==========================================
chosen_linkage = 'ward'

# ELBOW METHOD (Moons)
inertia_scores_moons = []
k_choices = range(1, 11)
for k in k_choices:
    # AgglomerativeClustering does not accept n_clusters=1, so we handle baseline inertia manually
    if k == 1:
        labels = np.zeros(X_moons.shape[0])
    else:
        km = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_moons)
        labels = km.labels_
    inertia_scores_moons.append(get_agglomerative_inertia(X_moons, labels))

kl_moons = KneeLocator(k_choices, inertia_scores_moons, curve='convex', direction='decreasing')
moons_elbow_k = kl_moons.elbow if kl_moons.elbow is not None else 2

# Gap Statistic (Moons)
gap_scores_moons = []
se_scores_moons = []  
B = 100  
X_min_moons = X_moons.min(axis=0)
X_max_moons = X_moons.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores_moons[idx]
    ref_log_inertias = []
    
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min_moons, X_max_moons, X_moons.shape)
        if k == 1:
            ref_labels = np.zeros(X_ref.shape[0])
        else:
            km_ref = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_ref)
            ref_labels = km_ref.labels_
        
        ref_inertia = get_agglomerative_inertia(X_ref, ref_labels)
        ref_log_inertias.append(np.log(ref_inertia) if ref_inertia > 0 else 0)    
    
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores_moons.append(gap)
    
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores_moons.append(se)

moons_gap_k = k_choices[-1]  
for i in range(len(k_choices) - 1):
    if gap_scores_moons[i] >= gap_scores_moons[i + 1] - se_scores_moons[i + 1]:
        moons_gap_k = k_choices[i]
        break

# Silhouette, DBI, CH Loop (Moons)
moons_best_silhouette_k = 2
best_sil_score = -1
moons_best_dbi_k = 2
best_dbi_score = float('inf')  
moons_best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    temp_model = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_moons)
    labels = temp_model.labels_
    
    sil_score = silhouette_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_sil_score = sil_score
        moons_best_silhouette_k = k

    dbi_score = davies_bouldin_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, moons_best_dbi_k = dbi_score, k

    ch_score = calinski_harabasz_score(X_moons, labels)
    print(f"Testing Moons k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, moons_best_ch_k = ch_score, k

print(f"Optimal clusters for Moons (Elbow): {moons_elbow_k}")
print(f"Optimal clusters for Moons (Gap): {moons_gap_k}")
print(f"Optimal clusters for Moons (Sil): {moons_best_silhouette_k}")
print(f"Optimal clusters for Moons (DBI): {moons_best_dbi_k}")
print(f"Optimal clusters for Moons (CH): {moons_best_ch_k}")

# Model Generation & Plotting Pipeline (Moons)
agglo_E = AgglomerativeClustering(n_clusters=moons_elbow_k, linkage=chosen_linkage).fit(X_moons)
agglo_G = AgglomerativeClustering(n_clusters=moons_gap_k, linkage=chosen_linkage).fit(X_moons)
agglo_S = AgglomerativeClustering(n_clusters=moons_best_silhouette_k, linkage=chosen_linkage).fit(X_moons)
agglo_D = AgglomerativeClustering(n_clusters=moons_best_dbi_k, linkage=chosen_linkage).fit(X_moons)
agglo_C = AgglomerativeClustering(n_clusters=moons_best_ch_k, linkage=chosen_linkage).fit(X_moons)

models = [agglo_E.labels_, agglo_S.labels_, agglo_D.labels_, agglo_C.labels_, agglo_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Moons Dataset Agglomerative Comparisons", fontsize=16, fontweight='bold')

for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    scatter = ax.scatter(X_moons[:, 0], X_moons[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{names[i]} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        true_species_inside = y_moons[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Class {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)
fig.tight_layout()
"""

# ==========================================
# 3. BLOBS DATASET PROCESSING (AGGLOMERATIVE)
# ==========================================

# ELBOW METHOD (Blobs)
inertia_scores_blobs = []
B = 100
k_choices = range(1, 11)
for k in k_choices:
    if k == 1:
        labels = np.zeros(X_blobs.shape[0])
    else:
        km = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_blobs)
        labels = km.labels_
    inertia_scores_blobs.append(get_agglomerative_inertia(X_blobs, labels))

kl_blobs = KneeLocator(k_choices, inertia_scores_blobs, curve='convex', direction='decreasing')
blobs_elbow_k = kl_blobs.elbow if kl_blobs.elbow is not None else 2

# Gap Statistic (Blobs)
gap_scores_blobs = []
se_scores_blobs = []  
X_min_blobs = X_blobs.min(axis=0)
X_max_blobs = X_blobs.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores_blobs[idx]
    ref_log_inertias = []
    
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min_blobs, X_max_blobs, X_blobs.shape)
        if k == 1:
            ref_labels = np.zeros(X_ref.shape[0])
        else:
            km_ref = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_ref)
            ref_labels = km_ref.labels_
        
        ref_inertia = get_agglomerative_inertia(X_ref, ref_labels)
        ref_log_inertias.append(np.log(ref_inertia) if ref_inertia > 0 else 0)    
    
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores_blobs.append(gap)
    
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores_blobs.append(se)

blobs_gap_k = k_choices[-1]  
for i in range(len(k_choices) - 1):
    if gap_scores_blobs[i] >= gap_scores_blobs[i + 1] - se_scores_blobs[i + 1]:
        blobs_gap_k = k_choices[i]
        break

# Silhouette, DBI, CH Loop (Blobs)
blobs_best_silhouette_k = 2
best_sil_score = -1
blobs_best_dbi_k = 2
best_dbi_score = float('inf')  
blobs_best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    temp_model = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_blobs)
    labels = temp_model.labels_
    
    sil_score = silhouette_score(X_blobs, labels)
    print(f"Testing Blobs k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_sil_score = sil_score
        blobs_best_silhouette_k = k

    dbi_score = davies_bouldin_score(X_blobs, labels)
    print(f"Testing Blobs k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, blobs_best_dbi_k = dbi_score, k

    ch_score = calinski_harabasz_score(X_blobs, labels)
    print(f"Testing Blobs k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, blobs_best_ch_k = ch_score, k

print(f"Optimal clusters for Blobs (Elbow): {blobs_elbow_k}")
print(f"Optimal clusters for Blobs (Gap): {blobs_gap_k}")
print(f"Optimal clusters for Blobs (Sil): {blobs_best_silhouette_k}")
print(f"Optimal clusters for Blobs (DBI): {blobs_best_dbi_k}")
print(f"Optimal clusters for Blobs (CH): {blobs_best_ch_k}")

# Model Generation & Plotting Pipeline (Blobs)
agglo_E = AgglomerativeClustering(n_clusters=blobs_elbow_k, linkage=chosen_linkage).fit(X_blobs)
agglo_G = AgglomerativeClustering(n_clusters=blobs_gap_k, linkage=chosen_linkage).fit(X_blobs)
agglo_S = AgglomerativeClustering(n_clusters=blobs_best_silhouette_k, linkage=chosen_linkage).fit(X_blobs)
agglo_D = AgglomerativeClustering(n_clusters=blobs_best_dbi_k, linkage=chosen_linkage).fit(X_blobs)
agglo_C = AgglomerativeClustering(n_clusters=blobs_best_ch_k, linkage=chosen_linkage).fit(X_blobs)

models = [agglo_E.labels_, agglo_S.labels_, agglo_D.labels_, agglo_C.labels_, agglo_G.labels_]
names = ['Elbow Method', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'Gap Statistic']

print(f"BLOBS: Total Clusters Generated: {num_clusters}")
print(f"BLOBS: Total Data Points: {num_samples}")
print(f"BLOBS: Cluster Spreads (Densities): {[round(s, 2) for s in surprise_stds]}")
unique_blobs, blob_counts = np.unique(y_blobs, return_counts=True)
for b_id, count in zip(unique_blobs, blob_counts):
    percentage = (count / num_samples) * 100
    print(f"  Blob {b_id}: {count} points ({percentage:.1f}%)")
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Blobs Dataset Agglomerative Comparisons", fontsize=16, fontweight='bold')

for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    scatter = ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{names[i]} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        true_species_inside = y_blobs[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Blob {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)
fig.tight_layout()
"""
# ==========================================
# 4. CIRCLES DATASET PROCESSING (AGGLOMERATIVE)
# ==========================================

# ELBOW METHOD (Circles)
inertia_scores_circles = []
for k in k_choices:
    if k == 1:
        labels = np.zeros(X_circles.shape[0])
    else:
        km = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_circles)
        labels = km.labels_
    inertia_scores_circles.append(get_agglomerative_inertia(X_circles, labels))

kl_circles = KneeLocator(k_choices, inertia_scores_circles, curve='convex', direction='decreasing')
circles_elbow_k = kl_circles.elbow if kl_circles.elbow is not None else 2

# Gap Statistic (Circles)
gap_scores_circles = []
se_scores_circles = []  
X_min_circles = X_circles.min(axis=0)
X_max_circles = X_circles.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores_circles[idx]
    ref_log_inertias = []
    
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min_circles, X_max_circles, X_circles.shape)
        if k == 1:
            ref_labels = np.zeros(X_ref.shape[0])
        else:
            km_ref = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_ref)
            ref_labels = km_ref.labels_
        
        ref_inertia = get_agglomerative_inertia(X_ref, ref_labels)
        ref_log_inertias.append(np.log(ref_inertia) if ref_inertia > 0 else 0)    
    
    gap = np.mean(ref_log_inertias) - np.log(real_inertia)
    gap_scores_circles.append(gap)
    
    sdk = np.std(ref_log_inertias, ddof=1) if len(ref_log_inertias) > 1 else 0
    se = sdk * np.sqrt(1 + 1.0 / B)
    se_scores_circles.append(se)

circles_gap_k = k_choices[-1]  
for i in range(len(k_choices) - 1):
    if gap_scores_circles[i] >= gap_scores_circles[i + 1] - se_scores_circles[i + 1]:
        circles_gap_k = k_choices[i]
        break

# Silhouette, DBI, CH Loop (Circles)
circles_best_silhouette_k = 2
best_sil_score = -1
circles_best_dbi_k = 2
best_dbi_score = float('inf')  
circles_best_ch_k = 2
best_ch_score = -1

for k in range(2, 11):
    temp_model = AgglomerativeClustering(n_clusters=k, linkage=chosen_linkage).fit(X_circles)
    labels = temp_model.labels_
    
    sil_score = silhouette_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_sil_score:
        best_sil_score = sil_score
        circles_best_silhouette_k = k

    dbi_score = davies_bouldin_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, circles_best_dbi_k = dbi_score, k

    ch_score = calinski_harabasz_score(X_circles, labels)
    print(f"Testing Circles k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, circles_best_ch_k = ch_score, k

print(f"Optimal clusters for Circles (Elbow): {circles_elbow_k}")
print(f"Optimal clusters for Circles (Gap): {circles_gap_k}")
print(f"Optimal clusters for Circles (Sil): {circles_best_silhouette_k}")
print(f"Optimal clusters for Circles (DBI): {circles_best_dbi_k}")
print(f"Optimal clusters for Circles (CH): {circles_best_ch_k}")

# Model Generation & Plotting Pipeline (Circles)
agglo_E = AgglomerativeClustering(n_clusters=circles_elbow_k, linkage=chosen_linkage).fit(X_circles)
agglo_G = AgglomerativeClustering(n_clusters=circles_gap_k, linkage=chosen_linkage).fit(X_circles)
agglo_S = AgglomerativeClustering(n_clusters=circles_best_silhouette_k, linkage=chosen_linkage).fit(X_circles)
agglo_D = AgglomerativeClustering(n_clusters=circles_best_dbi_k, linkage=chosen_linkage).fit(X_circles)
agglo_C = AgglomerativeClustering(n_clusters=circles_best_ch_k, linkage=chosen_linkage).fit(X_circles)

models = [agglo_E.labels_, agglo_S.labels_, agglo_D.labels_, agglo_C.labels_, agglo_G.labels_]
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Circles Dataset Agglomerative Comparisons", fontsize=16, fontweight='bold')

for i in range(5):
    ax = axes[i]
    current_labels = models[i]
    scatter = ax.scatter(X_circles[:, 0], X_circles[:, 1], c=current_labels, cmap='viridis', edgecolors='k', alpha=0.8)
    ax.set_title(f"{names[i]} (k={len(np.unique(current_labels))})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    unique_clusters = np.unique(current_labels)
    legend_labels = []
    for cluster_id in unique_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        true_species_inside = y_circles[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        breakdown_parts = [f"{count} Class {s_idx}" for s_idx, count in zip(unique_species, species_counts)]
        legend_labels.append(f"C{cluster_id} ({total_pts} pts: {' + '.join(breakdown_parts)})")
        
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8)
fig.tight_layout()
"""
plt.show()
