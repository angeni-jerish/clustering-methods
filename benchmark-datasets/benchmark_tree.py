from imports import *

df = pd.read_csv("benchmark_results.csv")
method_names = {
    "ari_silhouette": "Silhouette",
    "ari_gap": "Gap",
    "ari_elbow": "Elbow",
    "ari_dbi": "DBI",
    "ari_chi": "CH",
}

ari_cols = ["ari_silhouette", "ari_gap", "ari_elbow", "ari_dbi", "ari_chi"]

df["best_method"] = (df[ari_cols].idxmax(axis=1)).map(method_names)