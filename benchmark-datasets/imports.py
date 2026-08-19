import random as py_random  
from itertools import product #simpler loops
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from kneed import KneeLocator
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp 
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors,LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text