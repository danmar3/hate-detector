"""
Main script for running all tests
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import nlp516
import nlp516.model
import nlp516.lstm.word2vec_lstm
import nltk
import argparse
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
FLAGS = None


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
    if isinstance(string, pd.DataFrame) and FLAGS.to_latex:
        print(string)
        print(string.to_latex(float_format=lambda x: '%.3f' % x),
              file=file)
    else:
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
    ''' run kfold tests for stage 1 models '''
    dataset = get_dataset(language)
    results = list()
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        results_i = eval_models(
            classifiers=CLASSIFIERS, vectorizers=VECTORIZERS,
            dataset=get_subtask_dataset(data, task))
        results.append(results_i)
    results = sum(results)/len(results)
    return results


def run_lstm_character_tests(language, task):
    ''' run kfold tests for lstm+character models '''
    if language == 'spanish':
        dataset = nlp516.data.DevelopmentSpanishB()
        n_train_steps = 2
    elif language == 'english':
        dataset = nlp516.data.DevelopmentEnglishB()
        n_train_steps = 2

    results = list()
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        _, result_i = nlp516.lstm.character_lstm.run_experiment(
            raw_df=data, target=[task], n_train_steps=n_train_steps)
        results.append(result_i)

    results = pd.DataFrame(
        {('lstm', 'character'):
         pd.DataFrame(results).mean()[[
             'accuracy', 'f1', 'precision', 'recall']]
         }).transpose()
    return results


def run_lstm_word2vec_tests(language, task):
    ''' run kfold tests for lstm+word2vec models '''
    if language == 'spanish':
        dataset = nlp516.data.DevelopmentSpanishB()
        if task == 'AG':
            n_train_steps = 1
        else:
            n_train_steps = 1
        vectorizers = {
            'word2vec_tweets': nlp516.word2vec.SpanishTweets,
            'fasttext_tweets': nlp516.word2vec.SpanishTweetsFastText
        }
    elif language == 'english':
        dataset = nlp516.data.DevelopmentEnglishB()
        n_train_steps = 1
        vectorizers = {
            'word2vec_news': nlp516.word2vec.FakeNews,
            'word2vec_tweets': nlp516.word2vec.EnglishTweets,
            'word2vec_tweets(filtered)': nlp516.word2vec.EnglishTweetsFiltered,
            'fasttext_tweets': nlp516.word2vec.EnglishTweetsFastText,
            'fasttext_tweets(filtered)':
            nlp516.word2vec.EnglishTweetsFilteredFastText
            }

    results = {('lstm', name): list() for name in vectorizers}
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        # run experiment for each vectorizer
        for name, Vectorizer in vectorizers.items():
            vectorizer = Vectorizer()
            vectorizer.load()
            _, result_i = nlp516.lstm.word2vec_lstm.run_experiment(
                raw=data, target=[task], n_train_steps=n_train_steps,
                vectorizer=vectorizer)
            results[('lstm', name)].append(result_i)
    results = pd.DataFrame(
        {name: pd.DataFrame(kfolds).mean()[['accuracy', 'f1', 'precision',
                                            'recall']]
         for name, kfolds in results.items()}).transpose()
    return results


def run_stage1():
    ''' run stage one tests '''
    languages = ['english', 'spanish']
    tasks = ['HS', 'TR', 'AG']
    # ---------------- all tests -----------------
    for language in languages:
        dataset = get_dataset(language)
        for task in tasks:
            with open("results_{}_{}.txt".format(language, task),
                      "w") as result_file:
                print_results('-'*30 + '{} {}'.format(language, task) + '-'*30,
                              file=result_file)
                results = run_stage1_tests(language=language,
                                           task=task)
                print_results(results, file=result_file)


def run_all_tests():
    ''' run all tests '''
    languages = ['english', 'spanish']
    tasks = ['HS', 'TR', 'AG']
    # ---------------- all tests -----------------
    for language in languages:
        dataset = get_dataset(language)
        for task in tasks:
            with open("results_{}_{}.txt".format(language, task),
                      "w") as result_file:
                print_results('-'*30 + '{} {}'.format(language, task) + '-'*30,
                              file=result_file)
                if FLAGS.stage1 or FLAGS.all_tests:
                    results1 = run_stage1_tests(language=language,
                                                task=task)
                else:
                    results1 = None
                if FLAGS.lstm or FLAGS.all_tests:
                    results2 = run_lstm_word2vec_tests(language=language,
                                                       task=task)
                else:
                    results2 = None
                if FLAGS.all_tests:
                    results3 = run_lstm_character_tests(language=language,
                                                        task=task)
                else:
                    results3 = None
                results = [r for r in [results1, results2, results3]
                           if r is not None]
                if len(results) == 1:
                    results = results[0]
                else:
                    results = pd.concat(results)
                print_results(results, file=result_file)


def main():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--to_latex', action='store_true')
    parser.add_argument('--stage1', action='store_true')
    parser.add_argument('--all_tests', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.to_latex:
        print('printing to latex')
    if FLAGS.stage1:
        print('running stage 1 tests')
        run_stage1()
    else:
        print('FLAGS: {}'.format(FLAGS))
        run_all_tests()


if __name__ == '__main__':
    main()
