"""
Word2Vec for fake news
@author: Daniel L. Marino (marinodl@vcu.edu)
"""

import os
import getpass
import gensim
import numpy as np
from . import data as datalib
from gensim.models import Word2Vec

_PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
if getpass.getuser() == 'marinodl':
    TMP_FOLDER = '/data/marinodl/tmp/word2vec/vectorizer'
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
else:
    TMP_FOLDER = os.path.join(_PROJECT_FOLDER, 'tmp/word2vec/vectorizer')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)


def gensim_preprocess(dataset):
    documents = []
    for line in dataset:
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    return documents


class FakeNews(object):
    def __init__(self, model_dir=None, language='english'):
        assert language == 'english', 'spanish not supported at the moment'
        if model_dir is None:
            model_dir = os.path.join(TMP_FOLDER,
                                     'fake_news_{}'.format(language))
        self.model_dir = model_dir
        self.language = language

    def load_fakenews(self):
        if not hasattr(self, '_documents'):
            dataset = datalib.external.load_fakenews()
            dataset = dataset[dataset.language == self.language]
            self._documents = gensim_preprocess(dataset.text)
        return self._documents

    def init(self, size=300, window=10, min_count=2, workers=10,
             epochs=10):
        self.model = Word2Vec(
            size=size, window=window, min_count=min_count,
            workers=workers)
        # train
        documents = self.load_fakenews()
        self.model.build_vocab(documents)
        self.model.train(sentences=documents, total_examples=len(documents),
                         epochs=epochs)
        self.save()

    def save(self):
        print('saving {} model in {}'.format(self, self.model_dir))
        self.model.save(self.model_dir)

    def load(self):
        if os.path.exists(self.model_dir):
            self.model = Word2Vec.load(self.model_dir)
        else:
            self.init()

    def preprocess(self, series):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = datalib.Tokenizer(self.language)
        dataset = series.apply(datalib.remove_urls_map)
        dataset = dataset.apply(datalib.remove_numbers)
        dataset = dataset.apply(datalib.replace_punctuation)  #
        dataset = dataset.apply(self._tokenizer)
        dataset = dataset.apply(datalib.user_camelcase_map)
        dataset = dataset.apply(datalib.hashtag_camelcase_map)
        dataset = dataset.apply(datalib.to_lowercase)
        # dataset = dataset.apply(datalib.remove_words_with_numbers)
        dataset = dataset.apply(datalib.remove_punctuation)
        return dataset

    def fit(self, sentences, epochs=5, concatenate=False):
        """use sentences to train the model
        Args:
            sentences (type): raw sentences
        """
        if not hasattr(self, 'model'):
            self.load()
        documents = self.preprocess(sentences)
        documents = list(documents)
        if concatenate:
            # self.model.build_vocab(documents)
            raise NotImplementedError('use concatenate false')
        else:
            self.model.build_vocab(sentences=documents, update=True)
            self.model.train(
                sentences=documents, total_examples=len(documents),
                epochs=epochs)
            self.save()

    def transform(self, data_x, data_y, zero_padding=True):
        def sentence2vectarray(sentence):
            return [self.model.wv[word] for word in sentence
                    if word in self.model.wv]

        output_x = list()
        output_y = list()
        corpus = self.preprocess(data_x)
        # apply word2vec
        for sentence, label in zip(corpus, data_y):
            vect_array = sentence2vectarray(sentence)
            if vect_array:
                output_x.append(np.stack(vect_array))
                output_y.append(label)
        if zero_padding:
            max_len = max(len(sentence) for sentence in output_x)
            padded_output = list()
            for sentence in output_x:
                zeros = np.zeros([max_len - sentence.shape[0],
                                  sentence.shape[1]])
                padded_output.append(np.concatenate([sentence, zeros], axis=0))
            output_x = np.stack(padded_output, axis=0)
            output_y = np.stack(output_y, axis=0)
        return output_x, output_y
