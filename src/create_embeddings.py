"""
Load the dataset and create two kinds of representations:
1. One-hot representation
2. TF-IDF representation

Everything will be saved in the data/processed folder as a json file with the keys:
1. id
2. cuisine
3. ingredients
4. onehot_representation
5. tfidf_representation

The one-hot and tf-idf representations are saved as a sparse matrix in the data/processed folder as pickles.
"""
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Dict
from scipy.sparse.csr import csr_matrix


def save_embeddings_and_vocab(embeddings: csr_matrix, vocab: Dict, embedding_type: str):
    """Save the embeddings and the vocabulary to processed folder."""
    assert embedding_type in ["onehot", "tfidf"]
    assert isinstance(embeddings, csr_matrix)
    with open(f"data/processed/{embedding_type}-embedding.pkl", "wb") as outfile:
        pickle.dump(embeddings, outfile)
    with open(f"data/processed/{embedding_type}-vocab.json", "w") as outfile:
        json.dump(vocab, outfile)


def generate_representations(data: Dict, minimum_word_frequency: int = 5) -> Dict:
    """
    Generate the one-hot and tf-idf representations of the ingredients.
    The representations are saved in to individual files in the data/processed folder.
    A word is added to the vocabulary if it appears at least 5 times in the corpus.
    """
    # 1. Generate the corpus of all ingredients.
    updated_data = []
    corpus = []
    for i, entry in enumerate(data):
        entry_ingredients = [
            ingredient.replace(" ", "") for ingredient in entry["ingredients"]
        ]
        corpus.append(" ".join(entry_ingredients))
        entry["representation_idx"] = i
        updated_data.append(entry.copy())

    # 2. Create the count vectors.
    print("Creating the Count representation...")
    vectorizer = CountVectorizer(min_df=minimum_word_frequency)
    one_hot_representation = vectorizer.fit_transform(corpus)
    vocab = vectorizer.vocabulary_
    save_embeddings_and_vocab(one_hot_representation, vocab, "count")

    # 3. Create the tf-idf representation of the ingredients.
    print("Creating the tf-idf representation...")
    tfidf_vectorizer = TfidfVectorizer(min_df=minimum_word_frequency)
    tfidf_representation = tfidf_vectorizer.fit_transform(corpus)
    tfidf_vocab = tfidf_vectorizer.vocabulary_
    save_embeddings_and_vocab(tfidf_representation, tfidf_vocab, "tfidf")

    assert tfidf_representation.shape[0] == one_hot_representation.shape[0] == len(data)

    return updated_data


if __name__ == "__main__":
    from preprocessing import load_data

    raw_data = load_data(path="data/raw/train.json")

    processed_data = generate_representations(raw_data)

    print("Saving updated data...")
    PROCESSED = "data/processed/data.json"  # output file
    with open(PROCESSED, "w") as outfile:
        json.dump(processed_data, outfile)
