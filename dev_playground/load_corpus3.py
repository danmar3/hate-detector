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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

divider = 8000000

def main():
    print("hello")
    pass
    classifier, vector = train_cls()
    print("done training classifier")

    with open("corpus.txt") as inf:
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
                if count > 4000000:
                    break
            # print("--------- negatives  ------------")
            # for l in negatives:
            #     print(l)
            print("--------- positives ------------")
            for l in positives:
                print(l)
            print("positives/negatives", len(positives), len(negatives))
            print(count)


def predict_tweet(classifier, vectorizer, doc):
    vector = vectorizer.transform([doc])
        
    return classifier.predict(vector)

def get_train_data():
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
    return lines, labels, documents


def get_mixed_data():
    tr_lines, tr_labels, documents = get_train_data()
    n = len(tr_lines)

    corp_lines = []
    with open("corpus.txt") as inf:
        count = 0
        for line in inf:
            line = line.strip()
            count += 1

            if count == divider:
                print("reached divider")
            elif count > divider and count <= divider + n:
                corp_lines.append(line)
    lines = []
    labels = []
    for i in range(n):
        lines.append(tr_lines[i])
        labels.append(1)
        lines.append(corp_lines[i])
        labels.append(0)
    # lines = list(zip(tr_lines, corp_lines))
    # labels = list(zip([1]*n, [0]*n))
    # for i in range(100):
    #     print(lines[i])
    print(n, len(corp_lines))
    return lines, labels


def train_cls():
    lines, labels, documents = get_train_data()
    lines, labels = get_mixed_data()
    pass
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(lines).toarray()
    #wordsAndClassByTweet = list(zip(tweet_words, labels))
    #classifier = RandomForestClassifier()
    classifier = LogisticRegression()
    classifier.fit(X, labels)
    return classifier, vectorizer
    

  



if __name__ == "__main__":
    main()
    #get_mixed_data()