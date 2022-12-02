import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, OneHotEncoder

from config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from pca import PCAModule
from utils import read_json

# set numpy random seed
np.random.seed(42)

if __name__ == "__main__":

    # read data
    data = read_json(RAW_DATA_DIR / "train.json")

    labels = [i["cuisine"] for i in data]
    ingredients = [i["ingredients"] for i in data]

    # one hot encoding of ingredients
    label_binarizer = MultiLabelBinarizer()
    X = label_binarizer.fit_transform(ingredients)

    # convert list of labels to list of integers
    unique_labels = list(set(labels))

    pca = PCAModule(X)
    # X_pca = pca.pca_transform(400)
    unique_labels = list(set(labels))
    X_pca_2d = pca.pca_transform(2)

    kmeans = KMeans(n_clusters=len(unique_labels), random_state=42).fit(X)
    y_pred_kmeans = kmeans.labels_

    # gmm = GaussianMixture(n_components=len(unique_labels), random_state=42).fit(X)
    # y_pred_gmm = gmm.predict(X)

    # group all the predictions by predicted cuisine and show the 5 most frequent ingredients
    for i in range(len(unique_labels)):
        ingredients_in_cuisine = label_binarizer.inverse_transform(
            X[y_pred_kmeans == i]
        )
        ingredients_in_cuisine = [
            item for sublist in ingredients_in_cuisine for item in sublist
        ]
        ingredients_in_cuisine = pd.Series(ingredients_in_cuisine).value_counts()[:5]
        print(f"{unique_labels[i]}: {ingredients_in_cuisine}")

    # Visualize the clusters
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_pred_kmeans, s=50, cmap="viridis")
    ax.set_title("KMeans Clustering")
    plt.tight_layout()
    plt.show()

    a = 1
