"""
Main script for running all tests
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import nlp516
import nlp516.model
import nltk
import numpy as np
import pandas as pd
import itertools
import sklearn
import sklearn.tree
import sklearn.naive_bayes
import sklearn.ensemble
from types import SimpleNamespace
from sklearn.feature_extraction.text import CountVectorizer


def get_dataset(language):
    # ------------------------ load dataset -------------------- #
    if language == 'spanish':
        raw = nlp516.data.PublicSpanishDataset()
    elif language == 'english':
        raw = nlp516.data.PublicEnglishDataset()

    # ------------------------ preprocess dataset -------------------- #
    # TODO: add tokenizer specialised for spanish
    tokenizer_map = nlp516.data.Tokenizer('english')
    remove_stopwords_map = nlp516.data.RemoveStopWords(language)
    stemmer_map = nlp516.data.Stemmer(language)

    def preprocess(dataset):
        def run(data):
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.remove_urls_map)
            data = nlp516.data.map_column(
                data, 'text', tokenizer_map)
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.remove_user_map)
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.hashtag_camelcase_map)
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.to_lowercase)
            data = nlp516.data.map_column(
                data, 'text', remove_stopwords_map)
            data = nlp516.data.map_column(data, 'text', stemmer_map)
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.remove_words_with_numbers)
            data = nlp516.data.map_column(
                data, 'text', nlp516.data.remove_punctuation)
            return data
        return SimpleNamespace(train=run(dataset.train),
                               valid=run(dataset.valid))
    dataset = preprocess(raw)
    return dataset


def subtask_dataset(dataset, task):
    train = SimpleNamespace(x=dataset.train.text,
                            y=getattr(dataset.train, task))
    valid = SimpleNamespace(x=dataset.valid.text,
                            y=getattr(dataset.valid, task))
    return SimpleNamespace(train=train, valid=valid)


def eval_metrics(model, dataset):
    model.fit(dataset.train.x, dataset.train.y)
    return {'accuracy': model.score(dataset.valid.x, dataset.valid.y),
            'precision': model.precision_score(dataset.valid.x, dataset.valid.y),
            'recall': model.recall_score(dataset.valid.x, dataset.valid.y),
            'f1': model.f1_score(dataset.valid.x, dataset.valid.y)}


def instantiate_models(classifiers, vectorizers):
    models = {('MajorityBaseline', '-'): nlp516.model.MajorityBaseline()}
    models.update(
        {(c, v): nlp516.model.MlModel(classifier=classifiers[c](),
                                      vectorizer=vectorizers[v]())
         for c, v in itertools.product(classifiers.keys(), vectorizers.keys())
         }
    )
    return models


def eval_models(classifiers, vectorizers, task, dataset):
    models = instantiate_models(classifiers, vectorizers)
    results = {key: eval_metrics(model, dataset=subtask_dataset(dataset, task))
               for key, model in models.items()}
    return pd.DataFrame(results).transpose()


classifiers = {'linear': lambda: sklearn.linear_model.LogisticRegression(),
               'svm-linear': lambda: sklearn.svm.LinearSVC(),
               'svm-rbf': lambda: sklearn.svm.SVC(gamma='scale'),
               'tree': lambda: sklearn.tree.DecisionTreeClassifier(),
               'forest': lambda: sklearn.ensemble.RandomForestClassifier(),
               'bayes': lambda: sklearn.naive_bayes.GaussianNB()}
# vectorizers = {'frequency': lambda: nlp516.vectorizer.Unigram2(1000)}  #,
vectorizers = {'frequency': lambda: nlp516.vectorizer.Unigram(100),
               'presence': lambda: nlp516.vectorizer.UnigramPresence(100)}


def print_results(string, file):
    print(string)
    print(string, file=file)


def main():
    result_file = open("results.txt", "w")
    languages = ['spanish', 'english']
    tasks = ['HS', 'TR', 'AG']
    for language in languages:
        dataset = get_dataset(language)
        for task in tasks:
            print_results('-'*30 + '{} {}'.format(language, task) + '-'*30,
                          file=result_file)

            results = eval_models(
                classifiers=classifiers, vectorizers=vectorizers,
                task=task, dataset=dataset)
            print_results(results, file=result_file)


if __name__ == '__main__':
    main()
