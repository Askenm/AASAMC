import json
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def load_data(path='../data/raw/train.json')

    with open(path,'r') as file:
        data = json.load(file)

    return data



def select_features(unk_tokens,threshold=3):
    ic = Counter(unk_tokens)
    filtered = {k:v for k,v in dict(ic).items() if v >= threshold}
    return filtered



def transform_corpus(data,threshold):

    ids = []
    labels = []
    corpus = []
    unk_tokens = []
    for datapoint in data:
        unk_tokens+=datapoint['ingredients']

    filtered = select_features(unk_tokens,threshold=threshold)

    for datapoint in data:
        cont = True
        temp = []
        for ing in datapoint['ingredients']:
            temp.append(ing.replace(' ',''))
            unk_tokens.append(ing)
            if ing not in filtered.keys():
                cont = False
                break
        if not cont:
            continue
    
        ids.append(datapoint['id'])
        labels.append(datapoint['cuisine'])
        corpus.append(' '.join(temp))



    return labels,corpus





    
        


    
def vectorize_corpus(corpus,save_path='../data/processed/onehot_representation.pkl'):


    # create the transform
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(corpus)
    # summarize
    print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(corpus)
    # summarize encoded vector
    print(vector.shape)
    print(type(vector))
    print(vector.toarray())



    vector = np.c_[vector.T,labels]
    with open(save_path,'wb') as f:
        pickle.dump(vector, f)




def main():
    data = load_data()

    labels,corpus = transform_corpus(data,threshold = 3)


    vectorize_corpus(corpus,labels,save_path='../data/processed/onehot_representation.pkl')
