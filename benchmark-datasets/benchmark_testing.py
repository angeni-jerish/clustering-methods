from benchmark_class import BenchmarkDataset
from imports import *

scenarios = ["baseline", "high_dim", "high_overlap", "with_noise", "density_variation", "size_imbalance"]
seeds = [42, 100, 2026, 999]
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

for scenario in scenarios:
    levels = choose_knob_value(scenario)
    for d, knob, seed in product(dimensions, levels, seeds):
        # Override structural logic clash: high_dim forces feature count to track knob setting
        actual_d = knob if scenario == "high_dim" else d
        
        # Instantiate your optimized class
        dataset = BenchmarkDataset(
            scenario=scenario,
            n_clusters=10,
            n_features=int(actual_d),
            knob_value=knob,
            random_state=seed
        )
        
        # Pull outputs from your code
        meta = dataset.get_meta_features()
        results = dataset.km_methods()
        
        # Save every parameter and outcome to a line dictionary
        master_log.append({
            "scenario": scenario, "dimensions": actual_d, "knob": knob, "seed": seed,
            "hopkins": meta["hopkins"], "density_var": meta["density_variation"],
            "dist_concentration": meta["dist_concentration"], "outlier_rate": meta["outlier_rate"],
            "best_k_silhouette": results["k_silhouette"], "ari_silhouette": results["ari_silhouette"],
            "best_k_gap": results["k_gap"], "ari_gap": results["ari_gap"]
        })

# Flatten arrays straight to an analytical table spreadsheet
df_results = pd.DataFrame(master_log)
df_results.to_csv("benchmark_results.csv", index=False)

for scenario in scenarios:
    print(f"\n\n--- Running Benchmark for Scenario: {scenario} ---")
    test = BenchmarkDataset(scenario=scenario, n_clusters=10, n_features=10, n_samples=1000, knob_value=choose_knob_value(scenario), random_state=42)
    test.get_meta_features()
    test.km_methods()