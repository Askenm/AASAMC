import numpy as np

from gensim.models import Word2Vec

from config import RAW_DATA_DIR

from utils import read_json

# set numpy random seed
np.random.seed(42)

if __name__ == "__main__":

    # read data
    data = read_json(RAW_DATA_DIR / "train.json")

    labels = [i["cuisine"] for i in data]
    ingredients = [i["ingredients"] for i in data]

    # Word2vec
    model = Word2Vec(
        ingredients,
        min_count=1,
        vector_size=100,
        window=10,
        sg=1,
        workers=-1,
        alpha=0.025,
        min_alpha=0.0001,
        epochs=1000,
    )

    # save model
    model.save("food2vec.model")

    # load model
    model = Word2Vec.load("food2vec.model")

    a = 1
