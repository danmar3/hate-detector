"""Experiments with doc2vec
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
#import nlp516.data as data
import os
import zipfile
#import nlp516
import pandas
import numpy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#import gensim.downloader as api


def get_davidson_docs():
    with open('nlp516/dataset/davidson_labeled_data.csv', 'rb') as file:
        dataset = pandas.read_csv(file)#, sep='\t')

    documents = []
    for i in range(dataset.shape[0]):
        line = dataset.iloc[i].tweet
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    print("davidson length", len(documents))
    return documents


def get_sentiment140_docs():
    with open('nlp516/dataset/sentiment140.csv', 'rb') as file:
        dataset = pandas.read_csv(file)#, sep='\t')

    documents = []
    for i in range(dataset.shape[0]):
    #for i in range(10000):
        line = dataset.iloc[i].tweet
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep) 
    print("sentiment length", len(documents))
    return documents

def get_text_docs(filename, n):
    documents = []
    with open("corpus.txt") as inf:
        count = 0
        for line in inf:
            prep = gensim.utils.simple_preprocess(line)
            documents.append(prep) 
            count += 1
            if count > n:
                break
    return documents

def get_training_docs():
    with open('nlp516/dataset/development/train_en.tsv', 'rb') as file:
                dataset = pandas.read_csv(file, sep='\t')
    lines = []
    documents = []
    labels = []
    for i in range(dataset.shape[0]):
    #   print()
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
        #print(label)
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    return documents, labels



#########################################################################
import datetime

if __name__ == "__main__":
    print("hello")
    if input("Retrain model?(yes/no)") != "yes":
        exit()
    start = datetime.datetime.now()
    training_docs, labels = get_training_docs()

    corpus = []
    # corpus += get_sentiment140_docs()
    # corpus += get_davidson_docs()
    #corpus += get_text_docs("corpus.txt", 100000)
    corpus += get_text_docs("filtered_corpus.txt", 100000)
    #corpus += training_docs

    tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    print("training vectorizor....")
    model = Doc2Vec(tagged_docs, vector_size=300, window=2, min_count=1, workers=4)
    print("trained vectorizor")

    model.save("test.model")
    print("saved model")
    print("runtime", datetime.datetime.now()-start)

