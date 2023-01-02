#from mrmr import mrmr_classif

from sklearn.datasets import load_wine
import pandas as pd
from collections import defaultdict
import mifs

# http://lcsl.mit.edu/courses/regml/regml2017/slides/LeoLefakis.pdf
# https://github.com/danielhomola/mifs

wine = load_wine()

features, labels = wine.data, wine.target
data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
data_df["target"] = wine.target


# define MI_FS feature selection method
feat_selector = mifs.MutualInformationFeatureSelector(method='MRMR',k=25,n_features=13)

# find all relevant features
feat_selector.fit(features, labels)

# check selected features
feat_selector._support_mask

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(features)

features = wine.feature_names
ranked_features ={"rank": [], "feature": []}

unused = [i for i in range(13)]
for rank,ix in enumerate(feat_selector.ranking_):
    ranked_features['rank'].append(rank+1)
    ranked_features['feature'].append(features[ix])
    unused.pop(unused.index(ix))

    
ranked_features['rank'].append(rank+2)
ranked_features['feature'].append(features[12])
    
ranked_features = pd.DataFrame(ranked_features)


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
).sort_values('average_feature_information_rank')

outframe.to_csv("results/feature_information.csv", index=False)
outframe.to_latex("results/feature_information.tex",index=False)
ranked_features.to_latex("results/ranked_features.tex",index=False)