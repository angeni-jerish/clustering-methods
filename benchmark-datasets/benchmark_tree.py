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

print(df["best_method"].value_counts()) #ensuring every method has a sufficient number of "best runs"

feature_cols = [
    "hopkins","pca_pc1", "pca_dims_to_80",
    "pca_entropy", "outlier_rate", "density_var", "dimensions",
]
X = df[feature_cols]
y = df["best_method"]

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
tree.fit(X, y)

print(export_text(tree, feature_names=feature_cols))

for name, importance in sorted(zip(feature_cols, tree.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name}: {importance:.3f}")

leaf_ids = tree.apply(X)
df["leaf"] = leaf_ids
print(df.groupby("leaf")["best_method"].value_counts())
