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


def run_experiment(raw, name='test', target=['HS'], n_train_steps=10):
    # vectorize
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

    estimator = nlp516.lstm.lstm_model.BiLstmEstimator(
        num_inputs=train_x.shape[-1],
        num_units=[10, 10],
        keep_prob=SimpleNamespace(rnn=[None, 0.9],
                                  classifier=0.8),
        regularizer=0.0001,
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
