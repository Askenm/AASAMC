from mrmr import mrmr_classif
from sklearn.datasets import load_wine
import pandas as pd
from collections import defaultdict

# http://lcsl.mit.edu/courses/regml/regml2017/slides/LeoLefakis.pdf

wine = load_wine()

features, labels = wine.data, wine.target
data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
data_df["target"] = wine.target


feature_ranking = dict()
ranked = []

for i in range(1, 14):
    X = data_df[list(data_df.columns)[:-1]]
    y = data_df[list(data_df.columns)[-1]]
    selected_features = list(mrmr_classif(X=X, y=y, K=i))
    print(feature_ranking.values())
    print(selected_features)
    if ranked:
        for f in ranked:
            selected_features.pop(selected_features.index(f))

    feature_ranking[i] = selected_features[0]
    ranked += selected_features
    ranked = list(set(ranked))

ranked_features = pd.DataFrame(
    {"rank": feature_ranking.keys(), "feature": feature_ranking.values()}
)


weighted_sum = defaultdict(int)
dists_df = pd.read_csv("results/cluster_feature_distributions.csv")
for k, row in dists_df.iterrows():
    cur_rank = ranked_features.loc[ranked_features["feature"] == row["feature"]][
        "rank"
    ].values[0]
    weighted_sum[row["distribution"]] += cur_rank


for k, count in weighted_sum.items():
    weighted_sum[k] = count / dists_df.loc[dists_df.distribution == k].shape[0]


outframe = pd.DataFrame(
    {
        "distribution": list(weighted_sum.keys()),
        "average_feature_information_rank": list(weighted_sum.values()),
    }
)
outframe.to_csv("results/feature_information.csv", index=False)
outframe.to_latex("results/feature_information.tex",index=False)