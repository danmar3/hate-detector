"""
Split dataset in K fold
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import pandas as pd
import numpy as np
from types import SimpleNamespace


class KFold(object):
    def __init__(self, dataset, k):
        """Shuffles and splits the given dataset in k fold.
        Args:
            dataset (pd.Dataframe): pd.Dataframe to be shuffled and splitted.
            k (int): number of splits.
        """
        self.k = k
        dataset = pd.concat([dataset.train, dataset.valid])\
                    .reset_index(drop=True)
        shuffled = dataset.sample(frac=1)
        self.folds = np.array_split(shuffled, k)

    def __iter__(self):
        self._current_fold = 0
        return self

    def __next__(self):
        if self._current_fold < self.k:
            valid = self.folds[self._current_fold]
            train = [data for fold, data in enumerate(self.folds)
                     if fold != self._current_fold]
            self._current_fold += 1
            return SimpleNamespace(train=pd.concat(train),
                                   valid=valid)
        else:
            raise StopIteration()
