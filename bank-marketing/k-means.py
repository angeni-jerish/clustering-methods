import pandas as pd
from scipy.io import arff

import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np

data, meta = arff.loadarff('dataset_.csv')
df=pd.DataFrame(data)
df.head()
df.info()

df.columns = meta.names()
if df['Class'].dtype == object:
    df['Class'] = df['Class'].str.decode('utf-8').astype(int)

X = df.drop(columns=['Class'])
y = df['Class']

X_scaled = StandardScaler().fit_transform(X)

# ELBOW METHOD
inertia_scores = []
k_choices = range(2, 8) #there can be a maximum of 7 clusters in this dataset, as there are only 7 unique classes
for k in k_choices:
    km = KMeans(n_clusters=k).fit(X_scaled)
    inertia_scores.append(km.inertia_)
kl = KneeLocator(k_choices, inertia_scores, curve='convex', direction='decreasing')
auto_k_val = kl.elbow



#Gap Statistic
gap_scores = []
se_scores = []  # Array to store standard errors for each k
B = 100  # Number of random datasets to generate per k
X_min = X.min(axis=0)
X_max = X.max(axis=0)

for idx, k in enumerate(k_choices):
    real_inertia = inertia_scores[idx]
    ref_log_inertias = []
    
    # Generate reproducible uniform noise
    rng = np.random.default_rng(seed=int(k * 100))
    for b in range(B):
        X_ref = rng.uniform(X_min, X_max, X_scaled.shape)
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
gap_k_val = k_choices[-1]  # Fallback to max k if condition is never met

for i in range(len(k_choices) - 1):
    k_curr = k_choices[i]
    gap_curr = gap_scores[i]
    
    gap_next = gap_scores[i + 1]
    se_next = se_scores[i + 1]
    print(f"Testing k={k_curr} -> Gap Score: {gap_curr:.3f}")
    # Condition: Gap(k) >= Gap(k+1) - SE(k+1)
    if gap_curr >= gap_next - se_next:
        gap_k_val = k_curr
        break

#Sillhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index Loop
best_k = 2
best_score = -1
best_dbi_k = 2
best_dbi_score = float('inf')  
best_ch_k = 2
best_ch_score = -1

for k in range(2, 8):
    #Silhouette Score
    temp_model = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X_scaled)
    labels = temp_model.labels_
    sil_score = silhouette_score(X_scaled, labels)
    print(f"Testing k={k} -> Silhouette Score: {sil_score:.3f}")
    if sil_score > best_score:
        best_score = sil_score
        best_k = k

    #DBI
    dbi_score = davies_bouldin_score(X_scaled, labels)
    print(f"Testing k={k} -> Davies-Bouldin Index: {dbi_score:.3f}")
    if dbi_score < best_dbi_score: 
        best_dbi_score, best_dbi_k = dbi_score, k

    #CH
    ch_score = calinski_harabasz_score(X_scaled, labels)
    print(f"Testing k={k} -> Calinski-Harabasz Index: {ch_score:.3f}")
    if ch_score > best_ch_score: 
        best_ch_score, best_ch_k = ch_score, k

print(f"The optimal num of clusters (By Elbow method): {auto_k_val}")
print(f"The optimal num of clusters (By Gap Statistic method): {gap_k_val}")
print(f"The optimal num of clusters (By Silhouette Coefficient testing): {best_k}")
print(f"The optimal num of clusters (By Davies-Bouldin Index testing): {best_dbi_k}")
print(f"The optimal num of clusters (By Calinski-Harabasz Index testing): {best_ch_k}")

# K-Means Clustering Models
models = []
names = []
if auto_k_val is not None:
    kmeans_E = KMeans(n_clusters=auto_k_val, n_init=50, random_state=42).fit(X_scaled)
    models.append(kmeans_E)
    names.append('Elbow Method')
if best_k is not None:
    kmeans_S = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit(X_scaled)
    models.append(kmeans_S)
    names.append('Silhouette Coefficient')
if best_dbi_k is not None:
    kmeans_D = KMeans(n_clusters=best_dbi_k, n_init=50, random_state=42).fit(X_scaled)
    models.append(kmeans_D)
    names.append('Davies-Bouldin Index')
if best_ch_k is not None:
    kmeans_C = KMeans(n_clusters=best_ch_k, n_init=50, random_state=42).fit(X_scaled)
    models.append(kmeans_C)
    names.append('Calinski-Harabasz Index')
if gap_k_val is not None:
    kmeans_G = KMeans(n_clusters=gap_k_val, n_init=50, random_state=42).fit(X_scaled)
    models.append(kmeans_G)
    names.append('Gap Statistic')  


print("\n=== FINAL TECHNIQUE COMPARISON SCORECARD ===")
print(f"{'Technique':<20} | {'Chosen K':<8} | {'Max Sub Contrast':<18}")
print("-" * 55)

for name, model in zip(names, models):
    # Temporarily append cluster labels to calculate real sub contrast metrics
    df['Temp_Cluster'] = model.labels_
    
    # Calculate subscription rate (percentage of Class == 2) for each cluster
    cluster_sub_rates = df.groupby('Temp_Cluster')['Class'].apply(lambda c: (c == 2).mean() * 100)
    
    max_rate = cluster_sub_rates.max()
    min_rate = cluster_sub_rates.min()
    contrast = max_rate - min_rate
    
    print(f"{name:<20} | {model.n_clusters:<8} | {contrast:.2f}% ({min_rate:.1f}% to {max_rate:.1f}%)")

# Clean up dataframe column modification
df.drop(columns=['Temp_Cluster'], errors='ignore')
print("\nExecution complete.")