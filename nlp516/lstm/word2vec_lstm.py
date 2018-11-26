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
                                            raw.train[target].values)
    valid_x, valid_y = vectorizer.transform(raw.valid.text,
                                            raw.valid[target].values)

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
        estimator.fit(train_x.astype(np.float32),
                      train_y.astype(np.int32),
                      steps=150)
        estimator.evaluate(
            valid_x.astype(np.float32),
            valid_y.astype(np.int32))
    # Evaluate
    metrics = estimator.evaluate(
        valid_x.astype(np.float32),
        valid_y.astype(np.int32))
    metrics['f1'] = 2*((metrics['precision'] * metrics['recall']) /
                       (metrics['precision'] + metrics['recall']))
    return estimator, metrics


class LstmHateDetector(object):
    def __init__(self, name):
        self.vectorizer = nlp516.word2vec.FakeNews()
        self.vectorizer.load()
        self.model_dir = os.path.join(TMP_FOLDER, name)

    def _define_estimator(self, num_inputs):
        if os.path.exists(self.model_dir):
            print('removing existing model directory {}'
                  ''.format(self.model_dir))
            shutil.rmtree(self.model_dir)

        self.estimator = nlp516.lstm.lstm_model.AggregatedBiLstmEstimator(
            num_inputs=num_inputs,
            num_units=[10, 10],
            keep_prob=SimpleNamespace(rnn=[None, 0.9],
                                      classifier=0.9),
            regularizer=None,
            learning_rate=0.01,
            gradient_clip=0.5,
            batch_size=500,
            model_dir=self.model_dir
                  )

    def fit(self, train, valid, target=['HS']):
        self.vectorizer.fit(train.text, epochs=10)
        train_x, train_y = self.vectorizer.transform(train.text,
                                                     train[target].values)
        valid_x, valid_y = self.vectorizer.transform(valid.text,
                                                     valid[target].values)
        if not hasattr(self, 'estimator'):
            self._define_estimator(num_inputs=train_x.shape[-1])

        for i in range(2):
            self.estimator.fit(train_x.astype(np.float32),
                               train_y.astype(np.int32),
                               steps=150)
            self.estimator.evaluate(
                valid_x.astype(np.float32),
                valid_y.astype(np.int32))

    def evaluate(self, valid, target=['HS']):
        valid_x, valid_y = self.vectorizer.transform(valid.text,
                                                     valid[target].values)
        result = self.estimator.evaluate(
                valid_x.astype(np.float32),
                valid_y.astype(np.int32))
        return result

    def predict(self, sentence):
        test_sentence, _ = self.vectorizer.transform(pd.Series([sentence]),
                                                     [[None]])
        test_y = [y for y in self.estimator.predict(
            test_sentence.astype(np.float32))]
        return test_y[0]


class Ensamble(object):
    def __init__(self, models):
        self.models = models

    def predict(self, sentence):
        pred = [model.predict(sentence) for model in self.models]
        pred = pd.DataFrame(pd.DataFrame([r for r in pred])).mean()
        return pred


class EnsambleHateDetector(Ensamble):
    def __init__(self, k):
        self.models = [LstmHateDetector() for i in range(k)]

    def fit(self, train, valid):
        shuffled = train.sample(frac=1)
        folds = np.array_split(shuffled, len(self.models))
        for data, model in zip(folds, self.models):
            model.fit(train=data, valid=valid)

    def evaluate(self, valid):
        result = [model.evaluate(valid=valid)
                  for model in self.models]
        return result
