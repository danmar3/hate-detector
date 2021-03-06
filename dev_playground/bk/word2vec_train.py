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
            if count > n:
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



#########################################################################



def train_model():
    print("hello")
    if input("Retrain model?(yes/no)") != "yes":
        exit()
    start = datetime.datetime.now()
    training_docs, labels = get_training_docs()

    corpus = []
    #corpus += get_text_docs("corpus.txt", 100000)
    corpus += get_text_docs("filtered_corpus_mixed.txt", 210000)
    #corpus += training_docs

    print("corpus length:", len(corpus))
    print("training vectorizer....")
    model = gensim.models.Word2Vec(corpus, size=300, window=10, min_count=2, workers=4)
    model.train(corpus, total_examples=len(corpus), epochs=10)
    print("trained vectorizer")

    #model.save("word2vec.model")
    model.wv.save("w2v_wv.model")
    print("runtime", datetime.datetime.now()-start)

    print(model.wv.most_similar("woman"))
    print(model)

def load_model():
    #model = gensim.models.Word2Vec.load("word2vec.model")
    model = gensim.models.KeyedVectors.load("w2v_wv.model")
    return model

def vectorize(word, model):
    try:
        return model.word_vec(word)
    except KeyError:
        return [0.0] * model.vector_size

if __name__ == "__main__":
    train_model()