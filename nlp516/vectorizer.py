"""
Feature selection and vectorization of corpus
@author: Daniel L. Marino (marinodl@vcu.edu) (Primary author)
@author: Paul Hudgins (hudginpj@vcu.edu) (Secondary author)
"""
import uuid
import emoji
import string
import operator
import numpy as np
import nlp516.data
import gensim
import gensim.models.doc2vec


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


class Doc2Vec(object):
    def _preprocessing(self, data):
        corpus = list(map(lambda tags: ' '.join(tags), data))
        corpus = list(map(gensim.utils.simple_preprocess, corpus))
        return corpus

    def fit(self, data):
        data = self._preprocessing(data)
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate(data)]
        self.model = gensim.models.doc2vec.Doc2Vec(
            documents, vector_size=300, window=2, min_count=1, workers=4)

    def transform(self, data):
        data = self._preprocessing(data)
        vectors = [self.model.infer_vector(doc) for doc in data]
        x = np.stack(vectors, axis=0)
        return x


class CharacterVectorizer(object):
    START = uuid.uuid4()
    END = uuid.uuid4()
    EMOJI = uuid.uuid4()
    NUMBER = uuid.uuid4()
    UNKNOWN = uuid.uuid4()
    lower_case = string.ascii_lowercase
    upper_case = string.ascii_uppercase
    whitespace = string.whitespace
    punctuation = string.punctuation + '¿“”¡'
    known = set(lower_case + upper_case + whitespace + punctuation)
    known.update(set([START, END, EMOJI, NUMBER]))

    def _filter_characters(self, sentence):
        def replace_emoji(character):
            return (self.EMOJI if character in emoji.EMOJI_UNICODE.values()
                    else character)

        def replace_numbers(character):
            return (self.NUMBER if character in (str(i) for i in range(10))
                    else character)

        def replace_unknown(character):
            return (self.UNKNOWN if character not in self.known
                    else character)

        def replace(character):
            replace_list = [replace_emoji, replace_numbers, replace_unknown]
            for func in replace_list:
                character = func(character)
            return character

        return [replace(character) for character in sentence]

    def fit(self, data):
        vocabulary_keys = set([self.START, self.END])
        for sentence in data:
            vocabulary_keys.update(set(sentence))

        vocabulary_keys = set(self._filter_characters(vocabulary_keys))
        self.vocabulary = {
            character: idx for idx, character
            in enumerate(vocabulary_keys)}

    def transform(self, data):
        def unicode2sparse(record):
            return [(self.vocabulary[k] if k in self.vocabulary
                     else self.UNKNOWN)
                    for k in record]
        output = list()
        for raw in data:
            filtered = self._filter_characters(raw)
            record = [self.START] + filtered + [self.END]
            sparse = unicode2sparse(record)
            output.append(np.array(sparse))
        return output
