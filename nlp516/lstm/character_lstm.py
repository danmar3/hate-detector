"""
LSTM model for hate-detector using character representations
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import os
import nlp516
import pickle
import getpass
import nlp516
import nlp516.model
import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf


_PROJECT_FOLDER = os.path.dirname(nlp516.__file__)
if getpass.getuser() == 'marinodl':
    TMP_FOLDER = '/data/marinodl/tmp/nlp516/character_lstm'
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
else:
    TMP_FOLDER = os.path.join(_PROJECT_FOLDER, 'tmp/character_lstm')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
TMP_MODEL = os.path.join(TMP_FOLDER, 'model')
TMP_DATA = os.path.join(TMP_FOLDER, 'data')


def preprocess(raw):
    train = nlp516.data.map_column(
        raw.train, 'text', nlp516.data.remove_urls_map)
    valid = nlp516.data.map_column(
        raw.valid, 'text', nlp516.data.remove_urls_map)
    # remove very long tweets (probably outliers)
    train = train.loc[(train.text.apply(lambda x: len(x)) < 500), :]
    valid = valid.loc[(valid.text.apply(lambda x: len(x)) < 500), :]
    #
    train = nlp516.data.map_column(
        train, 'text', nlp516.data.remove_numbers)
    valid = nlp516.data.map_column(
        valid, 'text', nlp516.data.remove_numbers)
    # train = nlp516.data.map_column(
    #    train, 'text', nlp516.data.remove_punctuation)
    # valid = nlp516.data.map_column(
    #    valid, 'text', nlp516.data.remove_punctuation)
    train = nlp516.data.map_column(
        train, 'text', nlp516.data.to_lowercase)
    valid = nlp516.data.map_column(
        valid, 'text', nlp516.data.to_lowercase)
    raw = SimpleNamespace(train=train,
                          valid=valid)
    return raw


def vectorize_dataset(vectorizer, raw, filename=None,
                      ignore_cache=False):
    if filename is None:
        filename = os.path.join(TMP_DATA, 'preprocessed')
    if not os.path.exists(filename) or ignore_cache:
        train_x = vectorizer.transform(raw.train.text)
        train_y = np.expand_dims(raw.train.HS.values, 1)

        valid_x = vectorizer.transform(raw.valid.text)
        valid_y = np.expand_dims(raw.valid.HS.values, 1)

        lstm_dataset = SimpleNamespace(
            train=SimpleNamespace(x=train_x, y=train_y),
            valid=SimpleNamespace(x=valid_x, y=valid_y))
        folder_name = os.path.dirname(filename)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(filename, 'wb') as file:
            pickle.dump(lstm_dataset, file)
    with open(filename, 'rb') as file:
        return pickle.load(file)


def run_experiment(raw_df, name, n_train_steps=10):
    corpus = preprocess(raw_df)
    # 1. Vectorizer
    character_vect = nlp516.vectorizer.CharacterVectorizer()
    character_vect.fit(corpus.train.text)
    vectorizer = nlp516.vectorizer.StackedVectorizer(
        vectorizers=[character_vect,
                     nlp516.vectorizer.OneHotSequenceEncoder(
                         n_classes=len(character_vect.vocabulary),
                         time_major=False)])
    # 2. estimator
    estimator = nlp516.lstm.lstm_model.AggregatedBiLstmEstimator(
        num_inputs=len(character_vect.vocabulary),
        num_units=[50],
        keep_prob=SimpleNamespace(rnn=[None], classifier=None),
        regularizer=None,
        learning_rate=0.01,
        gradient_clip=0.5,
        batch_size=500,
        model_dir=TMP_MODEL + '_{}'.format(name))
    # 3. vectorize dataset
    dataset = vectorize_dataset(
        vectorizer=vectorizer, raw=corpus,
        filename=os.path.join(TMP_DATA, '{}_vectorized'.format(name)),
        ignore_cache=True
        )
    # 4. run training
    for i in range(n_train_steps):
        estimator.fit(dataset.train.x.astype(np.float32),
                      dataset.train.y.astype(np.int32),
                      steps=1000)
        estimator.evaluate(
            dataset.valid.x.astype(np.float32),
            dataset.valid.y.astype(np.int32))
    # 5. evaluate
    metrics = estimator.evaluate(
        dataset.valid.x.astype(np.float32),
        dataset.valid.y.astype(np.int32))
    metrics['f1'] = 2*((metrics['precision'] * metrics['recall']) /
                       (metrics['precision'] + metrics['recall']))
    return estimator, metrics


def main(language):
    if language == 'spanish':
        raw = nlp516.data.DevelopmentSpanishB()
    elif language == 'english':
        raw = nlp516.data.DevelopmentEnglishB()
    run_experiment(raw, 5, language)
