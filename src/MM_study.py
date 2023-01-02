from pomegranate import (
    GeneralMixtureModel,
    LogNormalDistribution,
    GammaDistribution,
    NormalDistribution,
)
import numpy as np
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

### FOLLOW git clone guide to install pomegranate
# pip did not work for me
# https://pomegranate.readthedocs.io/en/latest/install.html


distributions = {
    "lognormal": LogNormalDistribution,
    "gamma": GammaDistribution,
    "normal": NormalDistribution,
}
num_iters = 25


wine = load_wine()
features, labels = wine.data, wine.target
data_df = pd.DataFrame(wine.data, columns=wine.feature_names)
data_df["target"] = wine.target

results = {"distribution": [], "mutual_information": []}

for iteration in range(num_iters):

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels
    )

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    for name, distribution in distributions.items():

        model = GeneralMixtureModel.from_samples(
            distribution, n_components=3, X=X_train
        )
        model.fit(X_train)
        y_pred = model.predict(X_test)

        results["distribution"].append(name)
        results["mutual_information"].append(adjusted_mutual_info_score(y_test, y_pred))


pd.DataFrame(results).groupby("distribution").mean().sort_values(
    "mutual_information", ascending=False
).to_csv("results/mixture_models.csv")
