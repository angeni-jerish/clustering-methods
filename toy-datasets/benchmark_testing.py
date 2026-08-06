from clustering_methods import *

#Expecting 3 clusters, 2 features, 1000 samples
def benchmark_dataset(scenario, n_features=2, n_samples=1000, random_state=42):
    if scenario == "baseline":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, 
                          cluster_std=1.0, random_state=random_state)
        
    # 2. Dimensionality: Passed via n_features parameter
    elif scenario == "high_dim":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, 
                          cluster_std=1.0, random_state=random_state)
        
    # 3. Cluster Overlap: High standard deviation forces boundaries to merge
    elif scenario == "high_overlap":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, 
                          cluster_std=3.5, random_state=random_state)
        
    # 4. Synthetic Noise: Add completely random uniform noise features or rows
    elif scenario == "with_noise":
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, 
                          cluster_std=1.0, random_state=random_state)
        noise = np.random.uniform(low=X.min(), high=X.max(), size=(int(n_samples * 0.15), n_features))
        X = np.vstack([X, noise])
        y = np.concatenate([y, -1 * np.ones(noise.shape[0], dtype=int)]) # Noise marked as -1
        
    # 5. Density Variation: Clusters built with wildly different standard deviations
    elif scenario == "density_variation":
        X_1, y_1 = make_blobs(n_samples=n_samples//3, n_features=n_features, centers=[[0, 0]], cluster_std=0.5, random_state=random_state)
        X_2, y_2 = make_blobs(n_samples=n_samples//3, n_features=n_features, centers=[[10, 10]], cluster_std=2.5, random_state=random_state)
        X_3, y_3 = make_blobs(n_samples=n_samples//3, n_features=n_features, centers=[[-10, 10]], cluster_std=5.0, random_state=random_state)
        X = np.vstack([X_1, X_2, X_3])
        y = np.concatenate([y_1, y_2 + 1, y_3 + 2])
        
    # 6. Non-spherical Shapes: Non-linear geometry that challenges K-means' distance metric
    elif scenario == "shapes_moons":
        X, y = make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)
        
    # 7. Size Imbalance: Skewed population distribution per cluster
    elif scenario == "size_imbalance":
        X_1, y_1 = make_blobs(n_samples=int(n_samples * 0.90), n_features=n_features, centers=[[0, 0]], cluster_std=1.0, random_state=random_state)
        X_2, y_2 = make_blobs(n_samples=int(n_samples * 0.05), n_features=n_features, centers=[[8, 8]], cluster_std=1.0, random_state=random_state)
        X_3, y_3 = make_blobs(n_samples=int(n_samples * 0.05), n_features=n_features, centers=[[-8, 8]], cluster_std=1.0, random_state=random_state)
        X = np.vstack([X_1, X_2, X_3])
        y = np.concatenate([y_1, y_2 + 1, y_3 + 2])
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    # Mandatory step: Standardize data so scale differences don't mask metric behavior
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

