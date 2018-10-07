"""
Classifiers for hate speech
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import operator
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer(object):
    @staticmethod
    def get_corpus(dataset):
        return [' '.join(sample) for sample in dataset]

    def __init__(self, n_features):
        self.n_features = n_features
        self.vectorizer = CountVectorizer()

    def id2word(self, id):
        vocabulary_inv = {value: key for key, value
                          in self.vectorizer.vocabulary_.items()}
        return vocabulary_inv[id]

    def fit(self, data):
        data = self.get_corpus(data)
        x = self.vectorizer.fit_transform(data)
        count = {idx: c for idx, c in enumerate(x.toarray().sum(0))}
        count = sorted(count.items(), key=operator.itemgetter(1))
        self.features_idx = [c[0] for c in count[-self.n_features:]]

    def transform(self, data):
        data = self.get_corpus(data)
        x = self.vectorizer.transform(data)
        return x[:, self.features_idx]


class MlModel(object):
    def __init__(self, n_features):
        self.vectorizer = Vectorizer(n_features)
        self.classifier = sklearn.linear_model.LogisticRegression()

    def fit(self, x, y):
        self.vectorizer.fit(x)
        x_vect = self.vectorizer.transform(x)
        self.classifier.fit(x_vect, y)
        print('score: {}'.format(self.score(x, y)))

    def predict(self, x):
        x_vect = self.vectorizer.transform(x)
        return self.classifier.predict(x_vect)

    def score(self, x, y):
        x_vect = self.vectorizer.transform(x)
        test_y = self.classifier.predict(x_vect)
        return np.mean(test_y == y)
