import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


plt.style.use("fivethirtyeight")

wine = load_wine()

features, labels = wine.data, wine.target
data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
data_df["target"] = wine.target

# For each feature, for each cluster find the distribution that best fits the data
# K-means assumes a circular cluster layout, while GMM is generative and assumes a gaussian.
# HAC does not assume anything on the underlying function generating the data,
# therefore, it can deal with pretty complex layouts.

dfs = {}
fitter_objects = {}
dists = {"feature": [], "cluster_id": [], "distribution": [], "sumsquare_error": []}


for idx, col in enumerate(data_df.columns[:-1]):
    for cid in data_df["target"].unique():
        cluster_feature = data_df.loc[data_df["target"] == cid][col].values
        cluster_feature = cluster_feature.reshape(cluster_feature.shape[0], 1)
        print(cluster_feature.shape)
        scaler = StandardScaler()
        cluster_feature = scaler.fit_transform(cluster_feature)

        f = Fitter(
            cluster_feature, distributions=["norm", "gamma", "powerlaw", "lognorm"]
        )
        f.fit()

        best_dist = list(f.get_best(method="sumsquare_error").keys())[0]

        dfs[col] = f.summary()
        fitter_objects[col] = f

        dists["feature"].append(col)
        dists["cluster_id"].append(cid)
        dists["distribution"].append(best_dist)
        dists["sumsquare_error"].append(f.summary().iloc[0]["sumsquare_error"])


# We see that the most common distributions are the lognormal and gamma (which look super similar tbh.)
dists_df = pd.DataFrame(dists)
distribution_counts = pd.DataFrame(
    dists_df[["distribution", "cluster_id"]].value_counts()
)
distribution_counts.reset_index(inplace=True)


grouped_dict = defaultdict(list)

grouped_dict = defaultdict(list)
for cid in range(0, 3):
    cur_cluster = distribution_counts.loc[distribution_counts["cluster_id"] == cid]
    grouped_dict["cluster_id"].append(cid)
    for dist in ["norm", "gamma", "powerlaw", "lognorm"]:
        cur_dist = cur_cluster.loc[cur_cluster["distribution"] == dist]
        if cur_dist.shape[0] == 0:
            grouped_dict[dist].append(0)
        else:
            grouped_dict[dist].append(cur_dist[0].values[0])
distribution_frame = pd.DataFrame(grouped_dict).set_index("cluster_id")
distribution_frame.plot(kind="bar", figsize=[9, 9])
plt.legend(loc="upper left")

plt.savefig("visuals/cluster_distributions.png")
dists_df.to_csv("results/cluster_feature_distributions.csv", index=None)
dists_df.to_latex("results/cluster_feature_distributions.tex",index=False)