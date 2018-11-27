"""
LSTM model for hate-detector word vector representations
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import re
import os
import shutil
import nlp516
import nlp516.model
import nlp516.lstm.character_lstm
import nlp516.word2vec
import getpass
import gensim
import nltk
import numpy as np
import pandas as pd
import sklearn
from types import SimpleNamespace
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf


_PROJECT_FOLDER = os.path.dirname(nlp516.__file__)
if getpass.getuser() == 'marinodl':
    TMP_FOLDER = '/data/marinodl/tmp/nlp516/word2vec/lstm_model'
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
else:
    TMP_FOLDER = os.path.join(_PROJECT_FOLDER, 'tmp/word2vec/lstm_model')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)


def run_experiment(raw, name='test', target=['HS'], n_train_steps=10,
                   vectorizer=None):
    # vectorize
    if vectorizer is None:
        vectorizer = nlp516.word2vec.FakeNews()
        vectorizer.load()
    train_x, train_y = vectorizer.transform(raw.train.text,
                                            raw.train[target])
    valid_x, valid_y = vectorizer.transform(raw.valid.text,
                                            raw.valid[target])

    # estimator
    model_dir = os.path.join(TMP_FOLDER, name)
    if os.path.exists(model_dir):
        print('removing existing model directory {}'.format(model_dir))
        shutil.rmtree(model_dir)

    estimator = nlp516.lstm.lstm_model.AggregatedBiLstmEstimator(
        num_inputs=train_x.shape[-1],
        num_units=[50, 25],
        keep_prob=SimpleNamespace(rnn=[None, 0.9],
                                  classifier=0.9),
        regularizer=0.001,
        learning_rate=0.01,
        gradient_clip=0.5,
        batch_size=500,
        model_dir=model_dir
    )
    # Training
    for i in range(n_train_steps):
        estimator.fit(train_x.values.astype(np.float32),
                      train_y.values.astype(np.int32),
                      steps=150)
        estimator.evaluate(
            valid_x.values.astype(np.float32),
            valid_y.values.astype(np.int32))
    # Evaluate
    metrics = estimator.evaluate(
        valid_x.values.astype(np.float32),
        valid_y.values.astype(np.int32))
    metrics['f1'] = 2*((metrics['precision'] * metrics['recall']) /
                       (metrics['precision'] + metrics['recall']))
    return estimator, metrics


class LstmHateDetector(object):
    def __init__(self, name):
        # self.vectorizer = nlp516.word2vec.FakeNews()
        self.vectorizer = nlp516.word2vec.EnglishTweets()
        self.vectorizer.load()
        self.model_dir = os.path.join(TMP_FOLDER, name)

    def _define_estimator(self, num_inputs):
        if os.path.exists(self.model_dir):
            print('removing existing model directory {}'
                  ''.format(self.model_dir))
            shutil.rmtree(self.model_dir)

        self.estimator = nlp516.lstm.lstm_model.AggregatedBiLstmEstimator(
            num_inputs=num_inputs,
            num_units=[50, 25],
            keep_prob=SimpleNamespace(rnn=[None, 0.9],
                                      classifier=0.9),
            regularizer=0.001,
            learning_rate=0.01,
            gradient_clip=0.5,
            batch_size=500,
            model_dir=self.model_dir
                  )

    def fit(self, train, valid, target=['HS'], n_train_iter=2, steps=150):
        # self.vectorizer.fit(train.text, epochs=10)
        train_x, train_y = self.vectorizer.transform(train.text,
                                                     train[target])
        valid_x, valid_y = self.vectorizer.transform(valid.text,
                                                     valid[target])
        if not hasattr(self, 'estimator'):
            self._define_estimator(num_inputs=train_x.shape[-1])

        for i in range(n_train_iter):
            self.estimator.fit(train_x.values.astype(np.float32),
                               train_y.values.astype(np.int32),
                               steps=steps)
            self.estimator.evaluate(
                valid_x.values.astype(np.float32),
                valid_y.values.astype(np.int32))

    def evaluate(self, valid, target=['HS']):
        valid_x, valid_y = self.vectorizer.transform(valid.text,
                                                     valid[target])
        result = self.estimator.evaluate(
                valid_x.values.astype(np.float32),
                valid_y.values.astype(np.int32))
        return result

    def predict(self, test_data):
        if isinstance(test_data, str):
            test_data = pd.Series([test_data])
        embedded, _ = self.vectorizer.transform(test_data)
        test_y = [y for y in self.estimator.predict(
            embedded.astype(np.float32))]
        return pd.DataFrame(test_y, index=embedded.indexes['batch'].values)\
                 .applymap(lambda x: x[0])


class Ensamble(object):
    def __init__(self, models):
        self.models = models

    def predict(self, test_x):
        pred = [model.predict(test_x) for model in self.models]
        aggregated = sum(pred)/len(pred)
        aggregated['labels'] = \
            (aggregated[['probabilities']] > 0.5).astype(np.int32)
        return aggregated

    def accuracy_score(self, x, y):
        ''' accuracy of the model on predicting the labels for x'''
        pred = self.predict(x)
        return sklearn.metrics.accuracy_score(
            y_true=y[pred.index].values, y_pred=pred['labels'])

    def precision_score(self, x, y):
        pred = self.predict(x)
        return sklearn.metrics.precision_score(
            y_true=y[pred.index].values, y_pred=pred['labels'])

    def recall_score(self, x, y):
        pred = self.predict(x)
        return sklearn.metrics.recall_score(
            y_true=y[pred.index].values, y_pred=pred['labels'])

    def f1_score(self, x, y):
        pred = self.predict(x)
        return sklearn.metrics.f1_score(
            y_true=y[pred.index].values, y_pred=pred['labels'])


class EnsambleHateDetector(Ensamble):
    def __init__(self, k):
        self.models = [LstmHateDetector(name='fold_'.format(i))
                       for i in range(k)]

    def fit(self, train, valid):
        shuffled = train.sample(frac=1)
        folds = np.array_split(shuffled, len(self.models))
        for data, model in zip(folds, self.models):
            model.fit(train=data, valid=valid)


class EnsambleHateDetector2(Ensamble):
    def __init__(self, k):
        self.models = [LstmHateDetector(name='fold_'.format(i))
                       for i in range(k)]

    def fit(self, train, valid):
        for k, data in enumerate(nlp516.data.KFold(train, len(self.models))):
            self.models[k].fit(train=data.train, valid=valid,
                               n_train_iter=10)
