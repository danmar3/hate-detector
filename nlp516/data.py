import os
import re
import nltk
import nlp516
import zipfile
import pandas as pd
from nltk.corpus import stopwords

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
    #return nltk.word_tokenize(text)
    return nltk.tokenize.casual.casual_tokenize(text, strip_handles=False)
    
def remove_urls_map(text):
    return re.sub(r"http\S+", "", text)


def user_camelcase_map(tokens):
    ''' finds user tags and splits them following a CamelCase format '''
    def is_user(text):
        return (True if text[0]=='@'
                else False)

    def convert_camelcase(name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1@\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1@\2', s1)
        return [t for t in s2.split('@') if t]

    tokens = [(convert_camelcase(t) if is_user(t)
            else [t]) for t in tokens]
    tokens = [k for t in tokens for k in t]
    return tokens


def hashtag_camelcase_map(tokens):
    ''' finds hash tags and splits them following a CamelCase format '''
    def is_hashtag(text):
        return (True if text[0]=='#'
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

    
class Stemmer(object):
    def __init__(self, language):
        assert language in set(['english']),\
            'supported languages are: english'
        self.language = language
        self.stemmer = nltk.stem.PorterStemmer()
    
    def __call__(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
