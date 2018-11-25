print("starting")
import gensim
#import nlp516.data as data
import os
import zipfile
#import nlp516
print()
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

#divider = 8000000

def main(corpus_path, train_data_path, num_tweets):
    print("hello")
    pass
    classifier, vector = train_cls(corpus_path, train_data_path, num_tweets)
    print("done training classifier")

    with open(corpus_path) as inf:
        with open("filtered_corpus.txt", 'w') as outf:
            count = 0
            positives = []
            negatives = []
            for line in inf:
                count += 1
                
                pred = predict_tweet(classifier, vector, line)[0]
                #print("pred/line", pred, line)
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
            print("positives/negatives", len(positives), len(negatives))
            print(count)


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
    #   print()
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
        #print(label)
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    return lines, labels, documents


def get_mixed_data(corpus_path, train_data_path, num_tweets):
    tr_lines, tr_labels, tr_documents = get_train_data(train_data_path)
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
    #classifier = RandomForestClassifier()
    classifier = LogisticRegression()
    classifier.fit(X, labels)
    return classifier, vectorizer
    

  



if __name__ == "__main__":
    main('corpus.txt', '../nlp516/dataset/development/train_en.tsv', 9500000)
    #main('corpus_es.txt', '../nlp516/dataset/development/train_es.tsv', 3000)
    #get_mixed_data()