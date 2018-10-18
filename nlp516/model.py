"""
Classifiers for hate speech
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import numpy as np
import sklearn.linear_model
import sklearn.svm
import nlp516.vectorizer


class MlModel(object):
    ''' Model comprised of vectorizer and classifier '''
    def __init__(self, vectorizer, classifier):
        ''' Model for text classification comprised of a vectorizer
            and a classifier.
        Args:
            vectorizer: model that converts tokenized text into
                        vector representations
            classifier: model that classifies vectorized text into
                         a set of categories
        '''
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, x, y):
        ''' Fit the vectorizer and classifier models using training dataset.
        Args:
            x (nparray): training inputs comprised of list of tags.
            y (nparray): training labels.
        '''
        self.vectorizer.fit(x)
        x_vect = self.vectorizer.transform(x)
        self.classifier.fit(x_vect, y)
        print('score: {}'.format(self.score(x, y)))

    def predict(self, x):
        ''' Predict the labels for a set of features '''
        x_vect = self.vectorizer.transform(x)
        return self.classifier.predict(x_vect)

    def score(self, x, y):
        ''' accuracy of the model on predicting the labels for x'''
        test_y = self.predict(x)
        return np.mean(test_y == y)

    def precision_score(self, x, y):
        return sklearn.metrics.precision_score(
            y_true=y, y_pred=self.predict(x))

    def recall_score(self, x, y):
        return sklearn.metrics.recall_score(
            y_true=y, y_pred=self.predict(x))

    def f1_score(self, x, y):
        return sklearn.metrics.f1_score(
            y_true=y, y_pred=self.predict(x))


class MajorityBaseline(MlModel):
    ''' Model with Majority Baseline classifier '''
    def __init__(self):
        pass

    def fit(self, x, y):
        self.majority = (1.0 if np.mean(y) > 0.5
                         else 0.0)
        print('score: {}'.format(self.score(x, y)))

    def predict(self, x):
        out = np.zeros(shape=[x.shape[0]])
        out.fill(self.majority)
        return out


class SVMModel(MlModel):
    ''' Model with unigram (precense) vectorizer and SVC classifier '''
    def __init__(self, n_features):
        self.vectorizer = nlp516.vectorizer.UnigramPresence(n_features)
        self.classifier = sklearn.svm.SVC(gamma='scale')
