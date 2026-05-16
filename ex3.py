"""
Experiment3 : PCA and t-SNE on High Dimensional Dataset
Dataset: Breast Cancer Dataset (30 Features)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ==============================
# STEP 1: Load High-Dimensional Dataset
# ==============================

data = load_breast_cancer()
X = data.data      # 30 features
y = data.target    # target labels

print("Original Shape of Dataset:", X.shape)

# ==============================
# STEP 2: Standardize Data
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# STEP 3: Apply PCA
# ==============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Shape after PCA:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ==============================
# STEP 4: Apply t-SNE
# ==============================

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print("Shape after t-SNE:", X_tsne.shape)

# ==============================
# STEP 5: Visualization
# ==============================

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title("t-SNE Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

print("\nExperiment Completed Successfully!")