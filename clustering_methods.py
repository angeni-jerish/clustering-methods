import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors

#fixing issue with DBCV
if not hasattr(np, 'float_'):
    np.float_ = np.float64

# Load the Iris dataset
iris = load_iris() 
X = iris.data
flower_names = ['Setosa', 'Versicolor', 'Virginica']
# Standardize the features
X_scaled = StandardScaler().fit_transform(X)

#PCA (making 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# MOONS DATASET FOR DBSCAN
X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
moon_names = ['Bottom Crescent', 'Top Crescent']

''' 
#KEEP CODE IN CASE
for i, ax in enumerate(axes):
    current_labels = models[i]
    
    # Color points by the model's actual clusters
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=current_labels, cmap='viridis', edgecolors='k')
    ax.set_title(f"{names[i]}")
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")

    # Extract automatically generated handles for the legend
    handles, labels = scatter.legend_elements()
    legend_labels = []
    
    unique_clusters = np.unique(current_labels)
    # Isolate valid model clusters (ignoring DBSCAN noise label '-1' for the color legend elements)
    valid_clusters = unique_clusters[unique_clusters != -1]
    
    # Construct the tracking breakdown string for every cluster
    for cluster_id in valid_clusters:
        total_pts = np.sum(current_labels == cluster_id)
        
        # Cross-reference which real species fell into this model cluster
        true_species_inside = iris.target[current_labels == cluster_id]
        unique_species, species_counts = np.unique(true_species_inside, return_counts=True)
        
        breakdown_parts = []
        for s_idx, count in zip(unique_species, species_counts):
            breakdown_parts.append(f"{count} {flower_names[s_idx]}")
            
        species_breakdown = " + ".join(breakdown_parts)
        legend_labels.append(f"Cluster {cluster_id} ({total_pts} pts: {species_breakdown})")
        
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

    # Synchronize the handles to filter out the noise marker if it exists in legend handles
    if names[i] == 'DBSCAN' and -1 in unique_clusters:
        # Drop the first color handle (the purple noise handle) to align the rest
        ax.legend(handles[1:], legend_labels, title="Model Clusters Breakdown", loc="upper right", fontsize=7)
    else:
        ax.legend(handles, legend_labels, title="Model Clusters Breakdown", loc="upper right", fontsize=7)
'''