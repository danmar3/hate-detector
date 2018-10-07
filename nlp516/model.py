"""
Classifiers for hate speech
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import operator
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.svm
from sklearn.feature_extraction.text import CountVectorizer
import nlp516.vectorizer


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
    def __init__(self, vectorizer, classifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, x, y):
        self.vectorizer.fit(x)
        x_vect = self.vectorizer.transform(x)
        self.classifier.fit(x_vect, y)
        print('score: {}'.format(self.score(x, y)))

    def predict(self, x):
        x_vect = self.vectorizer.transform(x)
        return self.classifier.predict(x_vect)

    def score(self, x, y):
        test_y = self.predict(x)
        return np.mean(test_y == y)

    def precision_score(self, x, y):
        return sklearn.metrics.precision_score(
            y_true=y, y_pred=self.predict(x))

    def recall_score(self, x, y):
        return sklearn.metrics.recall_score(
            y_true=y, y_pred=self.predict(x))

    def f1_score(self, x, y):
        return sklearn.metrics.f1_score(
            y_true=y, y_pred=self.predict(x))


class MajorityBaseline(MlModel):
    def __init__(self):
        pass

    def fit(self, x, y):
        self.majority = (1.0 if np.mean(y) > 0.5
                         else 0.0)
        print('score: {}'.format(self.score(x, y)))

    def predict(self, x):
        out = np.zeros(shape=[x.shape[0]])
        out.fill(self.majority)
        return out


class SVMModel(MlModel):
    def __init__(self, n_features):
        self.vectorizer = nlp516.vectorizer.UnigramPresence(n_features)
        self.classifier = sklearn.svm.SVC(gamma='scale')
