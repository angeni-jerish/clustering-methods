from imports import *

class BenchmarkDataset:
    def __init__(self, scenario="baseline", n_clusters=10, n_features=2, n_samples=1000, knob_value=1.0,random_state=42):
        self.scenario = scenario
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.knob_value = knob_value
        self.X, self.y = self.choose_scenario()
        self.target = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto').fit(self.X)

    def choose_scenario(self):
        if self.scenario == "baseline":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            
        # 2. Dimensionality: Passed via n_features parameter
        elif self.scenario == "high_dim":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.knob_value, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            
        # 3. Cluster Overlap: High standard deviation forces boundaries to merge
        elif self.scenario == "high_overlap":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=self.knob_value, random_state=self.random_state)
            
        # 4. Synthetic Noise: Add completely random uniform noise features or rows
        elif self.scenario == "with_noise":
            X, y = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.n_clusters, 
                            cluster_std=1.0, random_state=self.random_state)
            noise = np.random.uniform(low=X.min(), high=X.max(), size=(int(self.n_samples * self.knob_value), self.n_features))
            X = np.vstack([X, noise])
            y = np.concatenate([y, -1 * np.ones(noise.shape[0], dtype=int)]) # Noise marked as -1
            
        elif self.scenario == "density_variation":
            #generated code, varied density by changing the standard deviation of each cluster
            X_list, y_list = [], []
            samples_per_cluster = self.n_samples // self.n_clusters
            
            for i in range(self.n_clusters):
                dynamic_std = 0.5 + (i * self.knob_value)
                angle = (2 * np.pi * i) / self.n_clusters
                
                separation_radius = 25.0 / np.sqrt(self.n_clusters)
                
                # Create a simple, flat list for the base coordinates
                center_coord = [separation_radius * np.cos(angle), separation_radius * np.sin(angle)]
                
                if self.n_features > 2:
                    padding = [0] * (self.n_features - 2)
                    center_coord = center_coord + padding

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
            size_dominant = int(self.n_samples * self.knob_value)  # knob_value should be between 0.1 and 0.9
            # Remaining clusters share the residual % evenly
            remaining_clusters = self.n_clusters - 1
            size_minority = int(self.n_samples * (1.0 - self.knob_value)) // remaining_clusters if remaining_clusters > 0 else 0
            
            for i in range(self.n_clusters):
                current_size = size_dominant if i == 0 else size_minority
                if current_size <= 0:
                    continue
                
                # Space out centers evenly across a geometric layout
                angle = (2 * np.pi * i) / self.n_clusters
                center_coord = [10 * np.cos(angle), 10 * np.sin(angle)]
                
                if self.n_features > 2:
                    padding = [0] * (self.n_features - 2)
                    center_coord = center_coord + padding
                center_coord = [center_coord]

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

    def get_meta_features(self, sampling_ratio=0.1):
        """Computes the five required data landscape properties."""
        rng = np.random.default_rng(self.random_state)
        n, d = self.X.shape

        # 1. Hopkins Statistic
        # Modified to run repeatedly across 20 different random samples
        hopkins_trials = []
        num_iterations = 20
        m = int(n * sampling_ratio)
        nn = NearestNeighbors(n_neighbors=2).fit(self.X)
        for _ in range(num_iterations):
            real_indices = rng.choice(n, size=m, replace=False)
            distances, _ = nn.kneighbors(self.X[real_indices], n_neighbors=2)
            w = distances[:, 1]
            min_bounds = self.X.min(axis=0)
            max_bounds = self.X.max(axis=0)
            X_uniform = rng.uniform(min_bounds, max_bounds, size=(m, d))
            u_distances, _ = nn.kneighbors(X_uniform, n_neighbors=1)
            u = u_distances[:, 0]

            sum_u, sum_w = np.sum(u), np.sum(w)
            trial_score = sum_u / (sum_u + sum_w) if (sum_u + sum_w) > 0 else 0.5
            hopkins_trials.append(trial_score)

        # The stable, final Hopkins Statistic
        hopkins = np.mean(hopkins_trials)

        # 2. Distance Concentration
        pairwise_dist = pdist(self.X, metric="euclidean")
        mean_dist = np.mean(pairwise_dist)
        dist_concentration = (
            np.std(pairwise_dist) / mean_dist if mean_dist > 0 else 0
        )

        # 3. PCA Spectrum (Full vector returned)
        pca = PCA().fit(self.X)
        pca_spectrum = pca.explained_variance_ratio_

        #split into 3 different sub characteristics
        # 1. Extracts the first Principal Component variance magnitude
        pca_pc1 = float(pca_spectrum[0]) if len(pca_spectrum) > 0 else 0.0

        # 2. Computes exactly how many components are needed to explain 80% of the shape
        pca_dims_to_80 = int(np.argmax(np.cumsum(pca_spectrum) >= 0.80) + 1)

        # 3. Computes the Shannon Entropy (describes the steepness of the curve)
        pca_entropy = -np.sum(pca_spectrum * np.log(pca_spectrum + 1e-9))

        # 4. Outlier Rate
        iso = IsolationForest(contamination="auto", random_state=self.random_state)
        outlier_rate = np.mean(iso.fit_predict(self.X) == -1)

        # 5. Local-Density Variation
        lof = LocalOutlierFactor(n_neighbors=min(20, n - 1))
        lof.fit_predict(self.X)
        local_densities = -lof.negative_outlier_factor_
        density_variation = np.var(local_densities)

        print(f"Hopkins Statistic: {hopkins:.3f}")
        print(f"Distance Concentration: {dist_concentration:.3f}")
        print(f"PCA Spectrum: {pca_spectrum}")
        print(f"Outlier Rate: {outlier_rate:.3f}")
        print(f"Local Density Variation: {density_variation:.3f}")


        return {
            "hopkins": hopkins,
            "dist_concentration": dist_concentration,
            "pca_pc1": pca_pc1,
            "pca_dims_to_80": pca_dims_to_80,
            "pca_entropy": pca_entropy,
            "outlier_rate": outlier_rate,
            "density_variation": density_variation
        }
    
    def km_methods(self):
        inertia_scores = []
        k_choices = range(1, 11)
        for k in k_choices:
            # Fixed: Standardised to ensure runtime acceleration constraints
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto').fit(self.X)
            inertia_scores.append(km.inertia_)

        try:
            kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
            auto_k_val = kl.elbow if kl.elbow is not None else 1
        except Exception:
            auto_k_val = 1 # Safe fallback if the data curve is unreadable shape noise

        # Gap Statistic
        gap_scores = []
        se_scores = []  
        B = 5  # Optimization: 20-50 reference arrays ensure massive speedups across grid swarms
        X_min = self.X.min(axis=0)
        X_max = self.X.max(axis=0)

        for idx, k in enumerate(k_choices):
            real_inertia = inertia_scores[idx]
            ref_log_inertias = []
            
            # Bound uniform generation states dynamically across seeds
            rng = np.random.default_rng(seed=int(self.random_state + k * 100))
            for b in range(B):
                X_ref = rng.uniform(X_min, X_max, self.X.shape)
                # Fixed: Dynamically mapped to track instance iteration states
                km_ref = KMeans(n_clusters=k, random_state=self.random_state, n_init='auto').fit(X_ref)
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
            # Fixed: Track instance random configurations natively
            temp_model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit(self.X)
            labels = temp_model.labels_
            sil_score = silhouette_score(self.X, labels)
            if sil_score > best_score:
                best_score = sil_score
                best_k = k
            # DBI
            dbi_score = davies_bouldin_score(self.X, labels)
            if dbi_score < best_dbi_score: 
                best_dbi_score, best_dbi_k = dbi_score, k

            # CH
            ch_score = calinski_harabasz_score(self.X, labels)
            if ch_score > best_ch_score: 
                best_ch_score, best_ch_k = ch_score, k

        print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
        print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
        print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
        print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
        print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")
        
        # K-Means Clustering Models Evaluated Direct via n_init='auto'
        ari_E, ari_S, ari_D, ari_C, ari_G = None, None, None, None, None

        if auto_k_val is not None:
            kmeans_E = KMeans(n_clusters=auto_k_val, n_init='auto', random_state=self.random_state).fit(self.X)
            ari_E = adjusted_rand_score(self.target.labels_, kmeans_E.labels_)
        if best_k is not None:
            kmeans_S = KMeans(n_clusters=best_k, n_init='auto', random_state=self.random_state).fit(self.X)
            ari_S = adjusted_rand_score(self.target.labels_, kmeans_S.labels_)
        if best_dbi_k is not None:
            kmeans_D = KMeans(n_clusters=best_dbi_k, n_init='auto', random_state=self.random_state).fit(self.X)
            ari_D = adjusted_rand_score(self.target.labels_, kmeans_D.labels_)
        if best_ch_k is not None:
            kmeans_C = KMeans(n_clusters=best_ch_k, n_init='auto', random_state=self.random_state).fit(self.X)
            ari_C = adjusted_rand_score(self.target.labels_, kmeans_C.labels_)
        if gap_k_val is not None:
            kmeans_G = KMeans(n_clusters=gap_k_val, n_init='auto', random_state=self.random_state).fit(self.X)
            ari_G = adjusted_rand_score(self.target.labels_, kmeans_G.labels_)

        return {
            "k_silhouette": best_k,
            "ari_silhouette": ari_S,
            "k_gap": gap_k_val,
            "ari_gap": ari_G,
            "k_elbow": auto_k_val,
            "ari_elbow": ari_E,
            "k_dbi": best_dbi_k,
            "ari_dbi": ari_D,
            "k_chi": best_ch_k,
            "ari_chi": ari_C
        }