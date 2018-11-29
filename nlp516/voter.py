"""
Split dataset in K fold
@authors: Viral Seth,
          Daniel L. Marino (marinodl@vcu.edu)
"""

import nlp516.data
K_FOLDS = 4


def run_experiment(train, test, language, task):
    # Run your code here
    results = {'accuracy': accuracy,
               'f1': f1,
               'precision': precision,
               'recall': recall}
    return results


def run_voter_main(language, task):
    if language == 'spanish':
        dataset = nlp516.data.DevelopmentSpanishB()
    elif language == 'english':
        dataset = nlp516.data.DevelopmentEnglishB()

    results = list()
    for k, data in enumerate(nlp516.data.KFold(dataset, K_FOLDS)):
        result_i = run_experiment(data.train, data.valid, language, task)
        results.append(result_i)
