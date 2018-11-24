"""
Dataset loading and preprocessing
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import os
import re
import nltk
import getpass
import nlp516
import string
import zipfile
import pandas as pd
from nltk.corpus import stopwords

_PROJECT_FOLDER = os.path.dirname(nlp516.__file__)
DATA_FOLDER = os.path.join(_PROJECT_FOLDER, 'dataset')

if getpass.getuser() == 'marinodl':
    TMP_FOLDER = '/data/marinodl/tmp/nlp516data/'
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
else:
    TMP_FOLDER = os.path.join(_PROJECT_FOLDER, 'tmp/dataset')
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)


def download_nltk_packets():
    nltk.download('stopwords')


class PublicTrialRaw(object):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/public_trial.zip')

    def __init__(self, src=None):
        archive = zipfile.ZipFile(self.src, 'r')
        with archive.open('trial_en.tsv') as file:
            self.en = pd.read_csv(file, sep='\t')
        with archive.open('trial_es.tsv') as file:
            self.es = pd.read_csv(file, sep='\t')


class PublicEnglishDataset(object):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development')

    def __init__(self, src=None):
        with open(os.path.join(self.src, 'train_en.tsv')) as file:
            self.train = pd.read_csv(file, sep='\t')
        with open(os.path.join(self.src, 'dev_en.tsv')) as file:
            self.valid = pd.read_csv(file, sep='\t')


class DevelopmentEnglishA(PublicEnglishDataset):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development_a/public_development_en')


class DevelopmentEnglishB(PublicEnglishDataset):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development_b/public_development_en')


class PublicSpanishDataset(object):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development')

    def __init__(self, src=None):
        with open(os.path.join(self.src, 'train_es.tsv')) as file:
            self.train = pd.read_csv(file, sep='\t')
        with open(os.path.join(self.src, 'dev_es.tsv')) as file:
            self.valid = pd.read_csv(file, sep='\t')


class DevelopmentSpanishA(PublicSpanishDataset):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development_a/public_development_es')


class DevelopmentSpanishB(PublicSpanishDataset):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/development_b/public_development_es')


def map_column(df, column, func):
    ''' applies the function func to a specific column '''
    def mapper(series):
        if series.name == column:
            return series.apply(func)
        else:
            return series
    return df.apply(mapper)


class Tokenizer(object):
    def __init__(self, language, tokenizer=None):
        self.language = language
        if tokenizer is None:
            self.tokenizer = nltk.tokenize.TweetTokenizer()
        else:
            self.tokenizer = tokenizer

    def __call__(self, text):
        return self.tokenizer.tokenize(text)


def casual_tokenize_map(text):
    # return nltk.word_tokenize(text)
    return nltk.tokenize.casual.casual_tokenize(text, strip_handles=False)


def remove_urls_map(text):
    return re.sub(r"http\S+", "", text)


def user_camelcase_map(tokens):
    ''' finds user tags and splits them following a CamelCase format '''
    def is_user(text):
        return (True if text[0] == '@'
                else False)

    def convert_camelcase(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1@\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1@\2', s1)
        return [t for t in s2.split('@') if t]

    tokens = [(convert_camelcase(t) if is_user(t)
               else [t]) for t in tokens]
    tokens = [k for t in tokens for k in t]
    return tokens


def remove_user_map(tokens):
    ''' finds user tags and remove them '''
    def is_user(text):
        return (True if text[0] == '@'
                else False)

    tokens = [t for t in tokens if not is_user(t)]
    return tokens


def hashtag_camelcase_map(tokens):
    ''' finds hash tags and splits them following a CamelCase format '''
    def is_hashtag(text):
        return (True if text[0] == '#'
                else False)

    def convert_camelcase(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s1)
        return [t for t in s2.split('#') if t]

    tokens = [(convert_camelcase(t) if is_hashtag(t)
               else [t]) for t in tokens]
    tokens = [k for t in tokens for k in t]
    return tokens


def to_lowercase(tokens):
    if isinstance(tokens, str):
        return tokens.lower()
    else:
        return [t.lower() for t in tokens]


class RemoveStopWords(object):
    def __init__(self, language):
        assert language in set(['english', 'spanish']),\
            'supported languages are: english, spanish'
        self.language = language
        self.stopwords = stopwords.words(self.language)

    def __call__(self, tokens):
        ''' Filter stop words '''
        return [word for word in tokens
                if word not in self.stopwords]


def remove_words_with_numbers(tokens):
    ''' Filter words with numbers '''
    return list(filter(lambda t: not any(c.isdigit() for c in t),
                       tokens))


def remove_numbers(sentence):
    remove_digits = str.maketrans('', '', string.digits)
    return sentence.translate(remove_digits)


def remove_punctuation(tokens):
    ''' remove punctuation from a list of tokens '''
    table = str.maketrans('', '', string.punctuation + '¿“”¡')
    if isinstance(tokens, str):
        return tokens.translate(table)
    else:
        stripped = [w.translate(table) for w in tokens]
        return [w for w in stripped if w]


def find_emojis(tokens):
    ''' find the emojis on a list of tokens '''
    return re.findall(r'[^\w\s,]', ' '.join(tokens))


class Stemmer(object):
    def __init__(self, language):
        assert language in set(['english', 'spanish']),\
            'supported languages are: english, spanish'
        self.language = language
        if self.language == 'english':
            self.stemmer = nltk.stem.PorterStemmer()
        elif self.language == 'spanish':
            self.stemmer = nltk.stem.SnowballStemmer("spanish")

    def __call__(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
