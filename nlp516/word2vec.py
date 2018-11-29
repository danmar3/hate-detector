"""
Word2Vec for fake news
@author: Daniel L. Marino (marinodl@vcu.edu)
"""

import os
import getpass
import gensim
import numpy as np
import pandas as pd
import xarray as xr
from . import data as datalib
from gensim.models import Word2Vec

_PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
if getpass.getuser() == 'marinodl':
    TMP_FOLDER = '/data/marinodl/tmp/nlp516/word2vec/vectorizer'
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    MODELS_FOLDER = '/data/marinodl/tmp/nlp516/models'
else:
    TMP_FOLDER = os.path.join(_PROJECT_FOLDER, 'tmp/word2vec/vectorizer')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    MODELS_FOLDER = '/dataset/models/'


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

    def init(self, size=200, window=10, min_count=2, workers=10,
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

    def get_word_embedding(self, word):
        # try:
        #     return self.model.wv.word_vec(word)
        # except KeyError:
        #     return None
        return (self.model.wv[word] if word in self.model.wv
                else None)

    def transform(self, data_x, data_y=None, zero_padding=True):
        assert isinstance(data_x, (pd.Series)), \
            'invalid data_x type {}'.format(type(data_x))
        if data_y is not None:
            assert isinstance(data_y, (pd.Series, pd.DataFrame))

        def sentence2vectarray(sentence):
            return [self.get_word_embedding(word) for word in sentence
                    if self.get_word_embedding(word) is not None]

        output_x = list()
        output_y = (list() if data_y is not None
                    else None)
        corpus = self.preprocess(data_x)
        batch_index = list()
        # apply word2vec
        for idx, sentence in enumerate(corpus):
            vect_array = sentence2vectarray(sentence)
            if vect_array:
                output_x.append(np.stack(vect_array))
                batch_index.append(corpus.index[idx])
                if data_y is not None:
                    output_y.append(data_y.iloc[idx, :])
        if zero_padding:
            max_len = max(len(sentence) for sentence in output_x)
            padded_output = list()
            for sentence in output_x:
                zeros = np.zeros([max_len - sentence.shape[0],
                                  sentence.shape[1]])
                padded_output.append(np.concatenate([sentence, zeros], axis=0))
            output_x = np.stack(padded_output, axis=0)
            output_x = xr.DataArray(
                output_x,
                coords=[batch_index,
                        pd.RangeIndex(0, output_x.shape[1]),
                        pd.RangeIndex(0, output_x.shape[2])],
                dims=['batch', 'time', 'embedding'])
            if data_y is not None:
                output_y = pd.DataFrame(output_y, index=batch_index)
        return output_x, output_y


class EnglishTweets(FakeNews):
    src = os.path.join(MODELS_FOLDER, 'wor2vec_raw_200k.model')

    def __init__(self):
        self.language = 'english'

    def get_word_embedding(self, word):
        try:
            return self.model.word_vec(word)
        except KeyError:
            return None

    def init(self):
        self.load()

    def load(self):
        self.model = gensim.models.KeyedVectors.load(self.src)

    def fit(self, *args, **kargs):
        pass

    def save(self):
        raise NotImplementedError('save not implemented for {}'.format(self))


class EnglishTweetsFiltered(EnglishTweets):
    src = os.path.join(MODELS_FOLDER, 'wor2vec_filtered_200k.model')


class EnglishTweetsFastText(EnglishTweets):
    src = os.path.join(MODELS_FOLDER, 'fasttext_filtered_200k.model')


class EnglishTweetsFilteredFastText(EnglishTweets):
    src = os.path.join(MODELS_FOLDER, 'fasttext_filtered_200k.model')


class SpanishTweets(EnglishTweets):
    src = os.path.join(MODELS_FOLDER, 'wor2vec_raw_200k_es.model')

    def __init__(self):
        self.language = 'spanish'
