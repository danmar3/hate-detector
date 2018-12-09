"""
Filters random language-specific tweets into a dataset-resembling corpus.
@author: Paul Hudgins (hudginspj@vcu.edu)
"""
import gensim
import os
import zipfile
import pandas
import numpy
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

CORPUS_FOLDER = 'dataset/corpuses/'
DATASET_FOLDER = 'dataset/development/'

def main(corpus_path, train_data_path, outfile, num_tweets):
    print("hello")
    pass
    classifier, vector = train_cls(CORPUS_FOLDER + corpus_path, DATASET_FOLDER + train_data_path, num_tweets)
    print("done training classifier")

    with open(CORPUS_FOLDER + corpus_path) as inf:
        with open(CORPUS_FOLDER + outfile, 'w') as outf:
            count = 0
            positives = []
            negatives = []
            for line in inf:
                count += 1
                pred = predict_tweet(classifier, vector, line)[0]
                if pred == 1:
                    positives.append(line)
                    outf.write(line)
                else:
                    negatives.append(line)
                if count % 10000 == 0:
                    print(count)
                if count >= num_tweets:
                    break
            # print("--------- negatives  ------------")
            # for l in negatives:
            #     print(l)
            # print("--------- positives ------------")
            # for l in positives:
            #     print(l)
            print("positives/negatives/total", len(positives), len(negatives), count)



def predict_tweet(classifier, vectorizer, doc):
    vector = vectorizer.transform([doc])    
    return classifier.predict(vector)

def get_train_data(train_data_path):
    with open(train_data_path, 'rb') as file:
            dataset = pandas.read_csv(file, sep='\t')

    lines = []
    documents = []
    labels = []
    for i in range(dataset.shape[0]):
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
    return lines, labels


def get_mixed_data(corpus_path, train_data_path, num_tweets):
    tr_lines, tr_labels = get_train_data(train_data_path)
    n = len(tr_lines)

    corp_lines = []
    with open(corpus_path) as inf:
        count = 0
        for line in inf:
            line = line.strip()
            count += 1

            if count == num_tweets:
                print("reached divider")
            elif count > num_tweets and count <= num_tweets + n:
                corp_lines.append(line)
    lines = []
    labels = []
    for i in range(n):
        lines.append(tr_lines[i])
        labels.append(1)
        lines.append(corp_lines[i])
        labels.append(0)
    print(n, len(corp_lines))
    return lines, labels


def train_cls(corpus_path, train_data_path, num_tweets):
    #lines, labels, documents = get_train_data()
    lines, labels = get_mixed_data(corpus_path, train_data_path, num_tweets)
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(lines).toarray()
    classifier = LogisticRegression()
    classifier.fit(X, labels)
    return classifier, vectorizer

if __name__ == "__main__":
    main('corpus_en.txt', 'train_en.tsv', 'filtered_corpus_en.txt', 9500000)
    main('corpus_es.txt', 'train_es.tsv', 'filtered_corpus_es.txt', 3500000)
