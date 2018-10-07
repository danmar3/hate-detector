"""
Feature selection and vectorization of corpus
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import operator
import numpy as np
import nlp516.data


class Unigram(object):
    def __init__(self, n_features):
        self.n_features = n_features

    def _preprocessing(self, data):
        corpus = list(map(nlp516.data.remove_punctuation, data))
        return corpus

    def _count_words(self, corpus):
        x_vec = np.zeros(shape=[len(corpus), len(self.vocabulary)])

        def unigram_map(k, tokens):
            # x = np.zeros(shape=[1, len(vocabulary)])
            idx = [self.vocabulary[w] for w in tokens
                   if w in self.vocabulary]
            for i in idx:
                x_vec[k, i] = x_vec[k, i] + 1

        list(map(lambda arg: unigram_map(*arg), enumerate(corpus)))
        return x_vec

    def id2word(self, id):
        vocabulary_inv = {value: key for key, value
                          in self.vocabulary.items()}
        return vocabulary_inv[id]

    def fit(self, data):
        corpus = self._preprocessing(data)
        self.vocabulary = {
            word: idx for idx, word
            in enumerate(set.union(*[set(sentence) for sentence in corpus]))}

        x_vec = self._count_words(corpus)
        count = {idx: c for idx, c in enumerate(x_vec.sum(0))}
        count = sorted(count.items(), key=operator.itemgetter(1))
        self.features_idx = [c[0] for c in count[-self.n_features:]]

    def transform(self, data):
        corpus = self._preprocessing(data)
        x_vec = self._count_words(corpus)
        return x_vec[:, self.features_idx]


class UnigramPresence(Unigram):
    def transform(self, data):
        corpus = self._preprocessing(data)
        x_vec = self._count_words(corpus)
        x_vec[x_vec > 1.0] = 1.0
        return x_vec[:, self.features_idx]


class Unigram2(Unigram):
    def fit(self, data):
        corpus = self._preprocessing(data)
        self.vocabulary = {
            word: idx for idx, word
            in enumerate(set.union(*[set(sentence) for sentence in corpus]))}

        x_vec = self._count_words(corpus)
        idx = [idx for idx, value in enumerate(x_vec.sum(0))
               if value > 10]

        vocabulary_inv = {value: key for key, value
                          in self.vocabulary.items()}
        self.vocabulary = {vocabulary_inv[old_idx]: new_idx
                           for new_idx, old_idx in enumerate(idx)}
        x_vec = self._count_words(corpus)
        count = {idx: c for idx, c in enumerate(x_vec.sum(0))}
        count = sorted(count.items(), key=operator.itemgetter(1))
        self.features_idx = [c[0] for c in count[:self.n_features]]
