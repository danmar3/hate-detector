"""
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
import os
import zipfile
import pandas
import numpy
import datetime

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

def get_training_docs():
    with open('../nlp516/dataset/development/train_en.tsv', 'rb') as file:
                dataset = pandas.read_csv(file, sep='\t')
    lines = []
    documents = []
    labels = []
    for i in range(dataset.shape[0]):
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
        #print(label)
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    return documents, labels

def train_fasttext(corpus):
    model = gensim.models.FastText(corpus, size=300, window=10, min_count=2, workers=4, iter=10)
    return model.wv

def train_word2vec(corpus):
    model = gensim.models.Word2Vec(corpus, size=300, window=10, min_count=2, workers=4)
    model.train(corpus, total_examples=len(corpus), epochs=10)
    return model.wv

def train_model(model_path, corp_path, n, train_func):
    start = datetime.datetime.now()
    # training_docs, labels = get_training_docs()

    corpus = get_text_docs(corp_path, n)
    print("corpus length:", len(corpus))

    model = train_func(corpus)
    model.save(model_path)

    print("runtime", (datetime.datetime.now()-start).total_seconds())
    print(model, model.most_similar("woman"))

def load_model(path):
    #model = gensim.models.Word2Vec.load("word2vec.model")
    model = gensim.models.KeyedVectors.load(path)
    return model

def vectorize(word, model):
    try:
        return model.word_vec(word)
    except KeyError:
        return [0.0] * model.vector_size

if __name__ == "__main__":
    if input(f"Retrain models? (yes/no) ") != "yes":
        exit()
    train_model("models/wor2vec_filtered_200k.model", "filtered_corpus_mixed.txt", 200000, train_word2vec)
    train_model("models/wor2vec_raw_200k.model", "corpus.txt", 200000, train_word2vec)
    train_model("models/fasttext_filtered_200k.model", "filtered_corpus_mixed.txt", 200000, train_fasttext)
    train_model("models/fasttext_raw_200k.model", "corpus.txt", 200000, train_fasttext)
    train_model("models/fasttext_raw_200k_es.model", "corpus_es.txt", 200000, train_fasttext)
    train_model("models/wor2vec_raw_200k_es.model", "corpus_es.txt", 200000, train_word2vec)
