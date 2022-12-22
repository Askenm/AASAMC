from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAModule:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.n_components: Union[int, None] = None

    def pca(self, n_components: int = 2) -> PCA:
        pca = PCA(n_components=n_components)
        pca.fit(self.data)
        return pca

    def pca_transform(self, n_components: int = 2) -> np.ndarray:
        pca = self.pca(n_components)
        return pca.transform(self.data)

    def plot(self, n_components: int = 2) -> None:
        pca = self.pca(n_components)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        plt.bar(
            range(0, len(exp_var_pca)),
            exp_var_pca,
            alpha=0.5,
            align="center",
            label="Individual explained variance",
        )
        plt.step(
            range(0, len(cum_sum_eigenvalues)),
            cum_sum_eigenvalues,
            where="mid",
            label="Cumulative explained variance",
        )
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal component index")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Make a random one-hot encoded matrix
    X = np.random.randint(2, size=(100, 100))

    # read pickle file
    X = pd.read_pickle("data/processed/onehot_representation.pkl")

    # make gaussian noise
    # X = X + np.random.normal(0, 0.1, X.shape)

    # Create a PCA object
    pca = PCAModule(X.todense())

    # Plot the explained variance
    pca.plot(1000)

    # Make a PCA object
    # pca = PCA(n_components=2)

    # # Fit the PCA object to the data
    # pca.fit(X)

    # # Transform the data
    # X_pca = pca.transform(X)

    # # Plot the data
    # plt.scatter(X_pca[:, 0], X_pca[:, 1])
    # plt.show()

    a = 1
