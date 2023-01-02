from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import Normalizer


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")


vis_path = "visuals/"

# import wine dataset from sklearn
wine = load_wine()
features, labels = wine.data, wine.target


# create dataframe from data
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target


# Make stratified train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# make a copy of the original data
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()


# Standardize the data
# Z-score scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize the data
# normalizer = Normalizer()
# X_train = normalizer.fit_transform(X_train)
# X_test = normalizer.transform(X_test)


score_frame = {
    "Clustering Method": [],
    "Adjusted Rand Index": [],
    "Adjusted Mutual Info Score": [],
}

# Clustering
# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)
score_frame["Clustering Method"].append("KMeans")
score_frame["Adjusted Rand Index"].append(adjusted_rand_score(y_test, y_pred))
score_frame["Adjusted Mutual Info Score"].append(
    adjusted_mutual_info_score(y_test, y_pred)
)


# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X_train)
y_pred = agg.fit_predict(X_test)
score_frame["Clustering Method"].append("Agglomerative Clustering")
score_frame["Adjusted Rand Index"].append(adjusted_rand_score(y_test, y_pred))
score_frame["Adjusted Mutual Info Score"].append(
    adjusted_mutual_info_score(y_test, y_pred)
)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
dbscan.fit(X_train)
y_pred = dbscan.fit_predict(X_test)
score_frame["Clustering Method"].append("DBSCAN Clustering")
score_frame["Adjusted Rand Index"].append(adjusted_rand_score(y_test, y_pred))
score_frame["Adjusted Mutual Info Score"].append(
    adjusted_mutual_info_score(y_test, y_pred)
)

# Gaussian Mixture
gm = GaussianMixture(n_components=3, random_state=42)
gm.fit(X_train)
y_pred = gm.predict(X_test)
score_frame["Clustering Method"].append("Gaussian Mixture")
score_frame["Adjusted Rand Index"].append(adjusted_rand_score(y_test, y_pred))
score_frame["Adjusted Mutual Info Score"].append(
    adjusted_mutual_info_score(y_test, y_pred)
)


evalution_df = pd.DataFrame(score_frame)
evalution_df.to_csv("results/evaluation.csv", index=None)


# Plot the data after PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# PCA the original data
X_train_orig_pca = pca.fit_transform(X_train_orig)
X_test_orig_pca = pca.transform(X_test_orig)

# Do the same for t-sne
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)
# X_test_tsne = tsne.transform(X_test)

# TSNE the original data
X_train_orig_tsne = tsne.fit_transform(X_train_orig)
# X_test_orig_tsne = tsne.transform(X_test_orig)


# Plot the data
fix, ax = plt.subplots(2, 2, figsize=(20, 10))

# Plot the data for each cluster
for i in range(3):
    ax[0, 0].scatter(
        X_train_pca[y_train == i, 0],
        X_train_pca[y_train == i, 1],
        label="Cluster {}".format(i),
    )
    ax[0, 1].scatter(
        X_train_orig_pca[y_train == i, 0],
        X_train_orig_pca[y_train == i, 1],
        label="Cluster {}".format(i),
    )
ax[0, 0].set_title("PCA of Wine Data (Standardized)")
ax[0, 1].set_title("PCA of Wine Data")
ax[0, 0].legend()
ax[0, 1].legend()


# Plot the data for each cluster
for i in range(3):
    ax[1, 0].scatter(
        X_train_tsne[y_train == i, 0],
        X_train_tsne[y_train == i, 1],
        label="Cluster {}".format(i),
    )
    ax[1, 1].scatter(
        X_train_orig_tsne[y_train == i, 0],
        X_train_orig_tsne[y_train == i, 1],
        label="Cluster {}".format(i),
    )
ax[1, 0].set_title("t-SNE of Wine Data (Standardized)")
ax[1, 1].set_title("t-SNE of Wine Data")
ax[1, 0].legend()
ax[1, 1].legend()

plt.savefig(vis_path + "plots.png")


# Calculate the performance of each clustering methods for each number of components using pca
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

methods = [KMeans, AgglomerativeClustering, GaussianMixture]
method_names = ["KMeans", "Agglomerative Clustering", "Gaussian Mixture"]
n_components = range(1, 13)
scores = []
for method, method_name in zip(methods, method_names):
    for n in n_components:
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        if method_name == "Agglomerative Clustering":
            cluster = method(n_clusters=3)
            y_pred = cluster.fit_predict(X_test_pca)
        elif method_name == "Gaussian Mixture":
            cluster = method(n_components=3, random_state=42)
            cluster.fit(X_train_pca)
            y_pred = cluster.predict(X_test_pca)
        else:
            cluster = method(n_clusters=3, random_state=42)
            cluster.fit(X_train_pca)
            y_pred = cluster.predict(X_test_pca)
        score = adjusted_rand_score(y_test, y_pred)
        scores.append([method_name, n, score])


# PCA: Plot the explained variance ratio
pca = PCA()
pca.fit(X_train)
fig, ax = plt.subplots(figsize=(16, 9))

plt.xlabel("Number of Components")
plt.ylabel("performance of clustering methods")

# plot the performance of each clustering methods for each number of components using pca
scores = pd.DataFrame(scores, columns=["method", "n_components", "score"])
import seaborn as sns


pca_frame = pd.DataFrame(
    {
        "n_components": range(1, 14),
        "Explained Variance": pca.explained_variance_ratio_.cumsum(),
    }
)
ax2 = plt.twinx()
ax2.set_yticks(
    np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
)
sns.lineplot(
    data=pca_frame["Explained Variance"],
    color="#6d904f",
    ax=ax2,
    marker="o",
    linestyle="--",
    alpha=0.95,
    linewidth=2.5,
)
sns.lineplot(
    x="n_components", y="score", hue="method", data=scores, alpha=0.65, linewidth=2.5
)
plt.legend(loc="lower left")

plt.ylim([0, 1.1])


plt.savefig(vis_path + "PCA_explained_variance.png")
