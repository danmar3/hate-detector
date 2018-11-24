"""
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
import os
import zipfile
import pandas
import numpy
import datetime
from word2vec_train import get_text_docs, get_training_docs


def train_model():
    print("hello")
    if input("Retrain model?(yes/no)") != "yes":
        exit()
    start = datetime.datetime.now()
    training_docs, labels = get_training_docs()

    corpus = []
    #corpus += get_text_docs("corpus.txt", 100000)
    corpus += get_text_docs("filtered_corpus_mixed.txt", 10000)
    #corpus += training_docs

    print("corpus length:", len(corpus))
    print("training vectorizer....")
    model = gensim.models.FastText(corpus, size=300, window=10, min_count=2, workers=4, iter=10)
    #model.train(corpus, total_examples=len(corpus), epochs=10)
    print("trained vectorizer")

    #model.save("word2vec.model")
    model.wv.save("fasttext_wv.model")
    print("runtime", datetime.datetime.now()-start)

    print(model.wv.most_similar("woman"))
    print(model)

def load_model():
    #model = gensim.models.Word2Vec.load("word2vec.model")
    model = gensim.models.KeyedVectors.load("bk/fasttext_wv.model")
    return model

def vectorize(word, model):
    try:
        return model.word_vec(word)
    except KeyError:
        return [0.0] * model.vector_size

if __name__ == "__main__":
    train_model()