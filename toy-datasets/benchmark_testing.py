from clustering_methods import *

class BenchmarkDataset:
    def __init__(self, scenario="baseline", n_clusters=10, n_features=10, n_samples=1000, random_state=42):
        self.scenario = scenario
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.X, self.y = self.choose_scenario()
        self.target = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.X)

    def get_target(self):
        return self.target
    
    #Expecting 3 clusters, 10 features, 1000 samples
    def choose_scenario(self):
        if self.scenario == "baseline":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            
        # 2. Dimensionality: Passed via n_features parameter
        elif self.scenario == "high_dim":
            X, y = make_blobs(n_samples=self.n_samples, n_features=20, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            
        # 3. Cluster Overlap: High standard deviation forces boundaries to merge
        elif self.scenario == "high_overlap":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=3.5, random_state=self.random_state)
            
        # 4. Synthetic Noise: Add completely random uniform noise features or rows
        elif self.scenario == "with_noise":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            noise = np.random.uniform(low=X.min(), high=X.max(), size=(int(self.n_samples * 0.15), self.n_features))
            X = np.vstack([X, noise])
            y = np.concatenate([y, -1 * np.ones(noise.shape[0], dtype=int)]) # Noise marked as -1
            
        elif self.scenario == "density_variation":
            X_list, y_list = [], []
            samples_per_cluster = self.n_samples // self.n_clusters
            
            for i in range(self.n_clusters):
                dynamic_std = 0.5 + (i * 1.5) 
                angle = (2 * np.pi * i) / self.n_clusters
                
                separation_radius = 25.0 / np.sqrt(self.n_clusters)
                
                # Create a simple, flat list for the base coordinates
                center_coord = [separation_radius * np.cos(angle), separation_radius * np.sin(angle)]
                
                if self.n_features > 2:
                    # FIX: Keep padding flat as a 1D list, then add them directly
                    padding = [0] * (self.n_features - 2)
                    center_coord = center_coord + padding

                # FIX: Wrap the completed 1D list inside an outer array layout to make it 2D
                center_coord = [center_coord]

                X_c, y_c = make_blobs(n_samples=samples_per_cluster, n_features=self.n_features, 
                                      centers=center_coord, cluster_std=dynamic_std, random_state=self.random_state + i)
                X_list.append(X_c)
                y_list.append(y_c + i) 
                
            X = np.vstack(X_list)
            y = np.concatenate(y_list)

        # 6. Size Imbalance: First cluster gets 90%, remaining points split equally among the rest
        elif self.scenario == "size_imbalance":
            X_list, y_list = [], []
            
            # Cluster 0 gets 90% dominance
            size_dominant = int(self.n_samples * 0.90)
            # Remaining clusters share the residual 10% evenly
            remaining_clusters = self.n_clusters - 1
            size_minority = int(self.n_samples * 0.10) // remaining_clusters if remaining_clusters > 0 else 0
            
            for i in range(self.n_clusters):
                current_size = size_dominant if i == 0 else size_minority
                if current_size <= 0:
                    continue
                
                # Space out centers evenly across a geometric layout
                angle = (2 * np.pi * i) / self.n_clusters
                center_coord = [[10 * np.cos(angle), 10 * np.sin(angle)]]
                
                if self.n_features > 2:
                    padding = [[0] * (self.n_features - 2)]
                    center_coord = [center_coord[0] + padding[0]]

                X_c, y_c = make_blobs(n_samples=current_size, n_features=self.n_features, 
                                      centers=center_coord, cluster_std=1.0, random_state=self.random_state + i)
                X_list.append(X_c)
                y_list.append(y_c + i)
                
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
        else:
            raise ValueError(f"Unknown scenario: {self.scenario}")
            
        # Standardize data so scale differences don't mask metric behavior
        X_scaled = StandardScaler().fit_transform(X)
        return X_scaled, y

    def km_methods(self):
        inertia_scores = []
        k_choices = range(1, 11)
        for k in k_choices:
            km = KMeans(n_clusters=k, random_state=self.random_state).fit(self.X)
            inertia_scores.append(km.inertia_)
        kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
        auto_k_val = kl.elbow

        # Gap Statistic
        gap_scores = []
        se_scores = []  
        B = 100  
        X_min = self.X.min(axis=0)
        X_max = self.X.max(axis=0)

        k_choices = range(1, 11)  # Testing k from 1 to 10
        for idx, k in enumerate(k_choices):
            real_inertia = inertia_scores[idx]
            ref_log_inertias = []
            
            # Generate reproducible uniform noise
            rng = np.random.default_rng(seed=int(k * 100))
            for b in range(B):
                X_ref = rng.uniform(X_min, X_max, self.X.shape)
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
            temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(self.X)
            labels = temp_model.labels_
            sil_score = silhouette_score(self.X, labels)
            print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
            if sil_score > best_score:
                best_score = sil_score
                best_k = k
            # DBI
            dbi_score = davies_bouldin_score(self.X, labels)
            print(f"Testing k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
            if dbi_score < best_dbi_score: 
                best_dbi_score, best_dbi_k = dbi_score, k

            # CH
            ch_score = calinski_harabasz_score(self.X, labels)
            print(f"Testing k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
            if ch_score > best_ch_score: 
                best_ch_score, best_ch_k = ch_score, k

        print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
        print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
        print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
        print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
        print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")


        # K-Means Clustering Models (Dynamically packed)
        models = []
        names = []
        if auto_k_val is not None:
            kmeans_E = KMeans(n_clusters=auto_k_val, n_init=50, random_state=42).fit(self.X)
            models.append(kmeans_E)
            names.append('Elbow Method')
            ari_E = adjusted_rand_score(self.target.labels_, kmeans_E.labels_)
            print(f"Adjusted Rand Index (ARI) for Elbow Method: {ari_E:.4f}")
        if best_k is not None:
            kmeans_S = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(self.X)
            models.append(kmeans_S)
            names.append('Silhouette Coefficient')
            ari_S = adjusted_rand_score(self.target.labels_, kmeans_S.labels_)
            print(f"Adjusted Rand Index (ARI) for Silhouette Coefficient: {ari_S:.4f}")
        if best_dbi_k is not None:
            kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=50, random_state=42).fit(self.X)
            models.append(kmeans_D)
            names.append('Davies-Bouldin Index')
            ari_D = adjusted_rand_score(self.target.labels_, kmeans_D.labels_)
            print(f"Adjusted Rand Index (ARI) for Davies-Bouldin Index: {ari_D:.4f}")
        if best_ch_k is not None:
            kmeans_C = KMeans(n_clusters=best_ch_k, n_init=50, random_state=42).fit(self.X)
            models.append(kmeans_C)
            names.append('Calinski-Harabasz Index')
            ari_C = adjusted_rand_score(self.target.labels_, kmeans_C.labels_)
            print(f"Adjusted Rand Index (ARI) for Calinski-Harabasz Index: {ari_C:.4f}")
        if gap_k_val is not None:
            kmeans_G = KMeans(n_clusters=gap_k_val, n_init=50, random_state=42).fit(self.X)
            models.append(kmeans_G)
            names.append('Gap Statistic')
            ari_G = adjusted_rand_score(self.target.labels_, kmeans_G.labels_)
            print(f"Adjusted Rand Index (ARI) for Gap Statistic Method: {ari_G:.4f}")

        
        # Terminal Profile Logging
        print(f"\nBLOBS: Total True Clusters Generated: {len(np.unique(self.y[self.y != -1]))}")
        print(f"BLOBS: Total Data Points: {len(self.y)}")
        unique_blobs, blob_counts = np.unique(self.y, return_counts=True)
        for b_id, count in zip(unique_blobs, blob_counts):
            percentage = (count / len(self.y)) * 100
            print(f"  Blob {b_id}: {count} points ({percentage:.1f}%)")

scenarios = ["baseline", "high_dim", "high_overlap", "with_noise", "density_variation", "size_imbalance"]

for scenario in scenarios:
    print(f"\n\n--- Running Benchmark for Scenario: {scenario} ---")
    test = BenchmarkDataset(scenario=scenario, n_clusters=10, n_features=10, n_samples=1000, random_state=42)
    test.km_methods()