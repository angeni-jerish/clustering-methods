from imports import *

df = pd.read_csv("benchmark_results.csv")
df = df.dropna(subset=["target_ari"])

method_names = {
    "win_S": "Silhouette",
    "win_G": "Gap",
    "win_E": "Elbow",
    "win_D": "DBI",
    "win_C": "CH",
}

win_cols = ["win_S", "win_G", "win_E", "win_D", "win_C"]

#checking if the fit is the same as the target when it identifies the true k
for method, k_col, win_col in [
    ("silhouette", "best_k_silhouette", "ari_silhouette"),
    ("gap", "best_k_gap", "ari_gap"),
    ("elbow", "best_k_elbow", "ari_elbow"),
    ("dbi", "best_k_dbi", "ari_dbi"),
    ("chi", "best_k_chi", "ari_chi"),
]:
    mask = df[k_col] == 10  # true k
    matches = df.loc[mask, win_col] == df.loc[mask, "target_ari"]
    print(f"{method}: {matches.sum()}/{mask.sum()} rows where best_k==10 match target_ari")

for win_col in win_cols:
    y = df[win_col]
    print(f"{method_names[win_col]} positive rate: {y.mean():.1%} ({y.sum()}/{len(y)})")

feature_cols = [
    "hopkins","dist_concentration", "pca_pc1", "pca_dims_to_80",
    "pca_entropy", "outlier_rate", "density_var", "dimensions",
]
X = df[feature_cols]
y = df[win_cols]

# One tree, fit on all 5 method outcomes at once -- each leaf can end up
# recommending zero, one, or several methods, instead of forcing a single winner.
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.125, random_state=42)
tree.fit(X, y)

print("\nFeature importances (combined across all 5 methods):")
for name, importance in sorted(zip(feature_cols, tree.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name}: {importance:.3f}")


def leaf_recommendation(node):
    tree_ = tree.tree_
    n = int(tree_.n_node_samples[node])
    scores = []
    for i, win_col in enumerate(win_cols):
        classes = tree.classes_[i]
        if True not in classes:
            continue  # this method never won a single row during training
        true_idx = list(classes).index(True)
        true_frac = tree_.value[node][i][true_idx]
        scores.append((method_names[win_col], true_frac))

    scores.sort(key=lambda t: -t[1])
    strong = [f"{name} ({frac:.0%})" for name, frac in scores if frac > 0.5]

    if strong:
        rec_str = ", ".join(strong)
    elif scores:
        # Nothing crossed 50% -- still name the best option available, so every
        # leaf gets a suggestion, but flag that it's a weak one.
        best_name, best_frac = scores[0]
        rec_str = f"{best_name} ({best_frac:.0%}, best available -- none crossed 50%)"
    else:
        rec_str = "no method has ever won here"

    return f"{rec_str}  (n={n})"


def print_tree(node=0, depth=0):
    tree_ = tree.tree_
    indent = "|   " * depth
    if tree_.children_left[node] == -1:
        print(f"{indent}|--- {leaf_recommendation(node)}")
    else:
        f = feature_cols[tree_.feature[node]]
        thr = tree_.threshold[node]
        print(f"{indent}|--- {f} <= {thr:.4f}")
        print_tree(tree_.children_left[node], depth + 1)
        print(f"{indent}|--- {f} >  {thr:.4f}")
        print_tree(tree_.children_right[node], depth + 1)


print("\nDecision tree (leaves show every method that wins there):")
print_tree()