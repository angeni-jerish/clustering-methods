from benchmark_class import BenchmarkDataset
from imports import *

scenarios = ["baseline", "high_dim", "high_overlap", "with_noise", "density_variation", "size_imbalance"]
seeds = [42, 100, 2026, 999, 7, 555, 31415, 8675309, 271828, 123456, 314, 2718, 1618, 90210, 12321, 55555, 777777, 4242, 8080, 999983]
dimensions = [2, 4, 8, 16, 32] 
overlap_levels = [1.0, 2.0, 3.0, 4.0, 5.0] 
noise_levels = [0.02, 0.08, 0.15, 0.25, 0.40] 
density_levels = [0.2, 0.6, 1.2, 1.8, 2.5] 
size_levels = [0.10, 0.35, 0.60, 0.85, 0.96]  

def choose_knob_value(scenario):
    if scenario == "high_dim":
        return dimensions  
    elif scenario == "high_overlap":
        return overlap_levels 
    elif scenario == "with_noise":
        return noise_levels  
    elif scenario == "density_variation":
        return density_levels 
    elif scenario == "size_imbalance":
        return size_levels  
    else:
        return [1.0]

master_log = []
#troubleshooting
total_runs = len(scenarios) * len(dimensions) * 5 * len(seeds) # rough estimation
current_run = 0
for scenario in scenarios:
    levels = choose_knob_value(scenario)

    for knob, seed in product(levels, seeds):
        current_run += 1
        print(f"\n[Run {current_run}] Processing -> Scenario: {scenario} | Knob: {knob} | Seed: {seed}")

        actual_d = knob if scenario == "high_dim" else 2
        # Instantiate your optimized class
        dataset = BenchmarkDataset(
            scenario=scenario,
            n_clusters=10,
            n_features=int(actual_d),
            knob_value=knob,
            random_state=seed
        )
        print(f"-> Calculating Target ARI...")
        target_ari = dataset.target_ari
        print(f"  -> Calculating Meta-features...")
        # Pull outputs from your code
        meta = dataset.get_meta_features()
        print(f"  -> Executing K-Means Val Indexes...")
        results = dataset.km_methods()
        
        # Save every parameter and outcome to a line dictionary
        master_log.append({
            "scenario": scenario, 
            "dimensions": actual_d, 
            "knob": knob, 
            "seed": seed,
            "target_ari" : target_ari,
            # Upfront Landscape Meta-Features
            "hopkins": meta["hopkins"], 
            "density_var": meta["density_variation"],
            "dist_concentration": meta["dist_concentration"], 
            "outlier_rate": meta["outlier_rate"],
            
            # --- ADD THE NEW PCA SUMMARY STATISTICS HERE ---
            "pca_pc1": meta["pca_pc1"],                 # Track linear flatness
            "pca_dims_to_80": meta["pca_dims_to_80"],   # Track structural complexity
            "pca_entropy": meta["pca_entropy"],         # Track variance scattering
            
            # Predicted K Choices
            "best_k_silhouette": results["k_silhouette"], 
            "best_k_gap": results["k_gap"], 
            "best_k_elbow": results["k_elbow"],
            "best_k_dbi": results["k_dbi"],
            "best_k_chi": results["k_chi"],
            
            # Clustering Accuracy (ARI) Columns
            "ari_silhouette": results["ari_silhouette"], 
            "ari_gap": results["ari_gap"],
            "ari_elbow": results["ari_elbow"],
            "ari_dbi": results["ari_dbi"],
            "ari_chi": results["ari_chi"],

            #Retrieved true k?
            "win_S": results["win_S"],
            "win_G": results["win_G"],
            "win_E": results["win_E"],
            "win_D": results["win_D"],
            "win_C": results["win_C"]
        })

# Flatten arrays straight to an analytical table spreadsheet
df_results = pd.DataFrame(master_log)
df_results.to_csv("benchmark_results.csv", index=False)
correlation_matrix = df_results.corr(numeric_only=True)

# Isolate just your accuracy rows (ARI) against your upfront shapes
print(correlation_matrix[["ari_silhouette", "ari_gap", "ari_elbow", "ari_dbi", "ari_chi"]])