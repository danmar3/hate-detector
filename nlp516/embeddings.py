"""
Trains fasttext and word2vec embeddings and provides tools for loading them
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
import os
import zipfile
import pandas
import numpy
import datetime


MODELS_FOLDER = 'dataset/models/'
CORPUS_FOLDER = 'dataset/corpuses/'

def get_text_docs(filename, n):
    documents = []
    with open(filename) as inf:
        count = 0
        for line in inf:
            prep = gensim.utils.simple_preprocess(line)
            documents.append(prep) 
            count += 1
            if count >= n:
                break
    return documents

def train_fasttext(corpus):
    model = gensim.models.FastText(corpus, size=300, window=10, min_count=2, workers=4, iter=10)
    return model

def train_word2vec(corpus):
    model = gensim.models.Word2Vec(corpus, size=300, window=10, min_count=2, workers=4)
    model.train(corpus, total_examples=len(corpus), epochs=10)
    return model

def train_model(model_path, corp_path, n, train_func):
    start = datetime.datetime.now()

    corpus = get_text_docs(CORPUS_FOLDER + corp_path, n)
    print("corpus length:", len(corpus))

    model = train_func(corpus)
    model.save(MODELS_FOLDER + model_path)
    print("Saved: ", model_path)

    print("runtime", (datetime.datetime.now()-start).total_seconds())
    #print(model, model.most_similar("woman"))

def load_model(path):
    #model = gensim.models.Word2Vec.load("word2vec.model")
    model = gensim.models.KeyedVectors.load(path)
    return model

def vectorize(word, model):
    try:
        return model.wv.word_vec(word)
    except KeyError:
        return [0.0] * model.vector_size


if __name__ == "__main__":
    if input(f"Retrain models? (yes/no) ") != "yes":
        exit()
    train_model("wor2vec_filtered_200k.model", "filtered_corpus_en.txt", 200000, train_word2vec)
    train_model("wor2vec_raw_200k.model", "corpus_en.txt", 200000, train_word2vec)
    train_model("fasttext_filtered_200k.model", "filtered_corpus_en.txt", 200000, train_fasttext)
    train_model("fasttext_raw_200k.model", "corpus_en.txt", 200000, train_fasttext)
    train_model("fasttext_raw_200k_es.model", "corpus_es.txt", 200000, train_fasttext)
    train_model("wor2vec_raw_200k_es.model", "corpus_es.txt", 200000, train_word2vec)
