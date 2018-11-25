"""
Main script for running all tests
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import nlp516
import nlp516.model
import nlp516.lstm.word2vec_lstm
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

K_FOLDS = 4


def get_dataset(language, run_preprocess=True):
    ''' load the training and validation dataset for a given language
    Args:
        language: 'spanish' or 'english'
        run_preprocess: run preprocessing
    Returns:
        dataset: struct with preprocessed training and validation datasets
    '''
    # ------------------------ load dataset -------------------- #
    if language == 'spanish':
        raw = nlp516.data.PublicSpanishDataset()
    elif language == 'english':
        raw = nlp516.data.PublicEnglishDataset()
    else:
        raise ValueError('Invalid language {}. Available languages are: '
                         '"Spanish", "English"'.format(language))

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
            if run_preprocess:
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


def get_subtask_dataset(dataset, task):
    ''' Extract dataset for a particular sub-task
    Args:
        dataset: english or spanish dataset with HS, TR and AG labels
        task (string): target task. Options are: 'HS', 'TR', 'AG'
    Returns:
        dataset: struct with training and validation datasets corresponding to
                 the given sub-task
    '''
    train = SimpleNamespace(x=dataset.train.text,
                            y=getattr(dataset.train, task))
    valid = SimpleNamespace(x=dataset.valid.text,
                            y=getattr(dataset.valid, task))
    return SimpleNamespace(train=train, valid=valid)


def train_and_evaluate(model, dataset):
    ''' Train model and Evaluate accuracy, precision, recal, and f1 scores for
        a given model on a given dataset'''
    model.fit(dataset.train.x, dataset.train.y)
    return {
        'accuracy': model.score(dataset.valid.x, dataset.valid.y),
        'precision': model.precision_score(dataset.valid.x, dataset.valid.y),
        'recall': model.recall_score(dataset.valid.x, dataset.valid.y),
        'f1': model.f1_score(dataset.valid.x, dataset.valid.y)}


def instantiate_models(classifiers, vectorizers):
    ''' Instantiate all models to being tested.
        Models are instantiated using a cartesian product between the list of
        classifiers and vectorizers '''
    models = {('MajorityBaseline', '-'): nlp516.model.MajorityBaseline()}
    models.update(
        {(c, v): nlp516.model.MlModel(classifier=classifiers[c](),
                                      vectorizer=vectorizers[v]())
         for c, v in itertools.product(classifiers.keys(), vectorizers.keys())
         }
    )
    return models


def eval_models(classifiers, vectorizers, dataset):
    ''' evaluate the performance of a set of classifiers and vectorizers on
        a given dataset '''
    models = instantiate_models(classifiers, vectorizers)
    results = {key: train_and_evaluate(model, dataset=dataset)
               for key, model in models.items()}
    return pd.DataFrame(results).transpose()


def eval_doc2vec(classifiers, dataset):
    ''' evaluate the performance of a set of classifiers on a given dataset
        using doc2vec vectorizer '''
    models = {
        (c, 'doc2vec'): nlp516.model.MlModel(
            classifier=classifiers[c](),
            vectorizer=nlp516.vectorizer.Doc2Vec())
        for c in classifiers.keys()
    }
    results = {key: train_and_evaluate(model, dataset=dataset)
               for key, model in models.items()}
    return pd.DataFrame(results).transpose()


def print_results(string, file):
    ''' print results into stdout and results file '''
    print(string)
    print(string, file=file)


CLASSIFIERS = {
    'linear': lambda: sklearn.linear_model.LogisticRegression(),
    'svc-linear': lambda: sklearn.svm.LinearSVC(),
    'svc-rbf': lambda: sklearn.svm.SVC(gamma='scale'),
    'tree': lambda: sklearn.tree.DecisionTreeClassifier(),
    'forest': lambda: sklearn.ensemble.RandomForestClassifier(),
    'bayes': lambda: sklearn.naive_bayes.GaussianNB(),
    'linear-sgd': lambda: sklearn.linear_model.SGDClassifier(),
    'gradient-boosted': lambda: sklearn.ensemble.GradientBoostingClassifier(),
    # 'NuSVC-rbf': lambda: sklearn.svm.NuSVC(gamma='scale', nu=0.1),
    # 'NuSVC-linear': lambda: sklearn.svm.NuSVC(gamma='scale', nu=0.1,
    #                                           kernel='linear')
}

VECTORIZERS = {
    'frequency': lambda: nlp516.vectorizer.Unigram(100),
    'presence': lambda: nlp516.vectorizer.UnigramPresence(100)
}


def run_stage1_tests(language, task):
    dataset = get_dataset(language)
    results = list()
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        results_i = eval_models(
            classifiers=CLASSIFIERS, vectorizers=VECTORIZERS,
            dataset=get_subtask_dataset(data, task))
        results.append(results)
    return results


def run_lstmword2vec_tests(language, task):
    if language == 'spanish':
        dataset = nlp516.data.DevelopmentSpanishB()
    elif language == 'english':
        dataset = nlp516.data.DevelopmentEnglishB()

    results = list()
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        _, result_i = nlp516.lstm.word2vec_lstm.run_experiment(
            raw=data, target=[task], n_train_steps=5)
        results.append(result_i)
    results = pd.DataFrame(
        {'lstm+word2vec':
         pd.DataFrame(results).mean()[[
             'accuracy', 'precision', 'recall', 'f1']]
         }).transpose()
    return results


def main():
    ''' run all tests '''
    result_file = open("results.txt", "w")
    languages = ['spanish', 'english']
    tasks = ['HS', 'TR', 'AG']
    # ---------------- unigram tests -----------------
    for language in languages:
        dataset = get_dataset(language)
        for task in tasks:
            print_results('-'*30 + '{} {}'.format(language, task) + '-'*30,
                          file=result_file)

            results = eval_models(
                classifiers=CLASSIFIERS, vectorizers=VECTORIZERS,
                dataset=get_subtask_dataset(dataset, task))
            print_results(results, file=result_file)

    # ---------------- doc2vec tests -----------------
    dataset = get_dataset('english', run_preprocess=False)
    for task in tasks:
        print_results('-'*30 + '{} {}'.format('english', task) + '-'*30,
                      file=result_file)
        results = eval_doc2vec(
            classifiers=CLASSIFIERS,
            dataset=get_subtask_dataset(dataset, task)
        )
        print_results(results, file=result_file)


if __name__ == '__main__':
    main()
