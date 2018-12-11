# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:52:23 2018

@author: vsheth
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

TRAINING_EN_DATA = r'C:\Users\vsheth\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_en_Task A\public_development_en_train_en.tsv'
TEST_EN_DATA = r'C:\Users\vsheth\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_en_Task A\public_development_en_dev_en.tsv'
TRAINING_ES_DATA = r'C:\Users\vsheth\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_Spanish_Task A\public_development_es_train_es.tsv'
TEST_ES_DATA = r'C:\Users\vsheth\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_Spanish_Task A\public_development_es_dev_es.tsv'

"""
TRAINING_EN_DATA = r'../hate-detector/nlp516/dataset/development/tra_en.tsv'
TEST_EN_DATA = r'../hate-detector/nlp516/dataset/development/dev_en.tsv'
TRAINING_ES_DATA = r'../hate-detector/nlp516/dataset/development/tra_es.tsv'
TEST_ES_DATA = r'../hate-detector/nlp516/dataset/development/dev_es.tsv'
"""

log_classifier = SklearnClassifier(LogisticRegression())
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SVCClassifier_classifier = SklearnClassifier(LinearSVC())
NuSVC_classifier = SklearnClassifier(NuSVC())
    
class Voter(ClassifierI):
    def __init__ (self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)    

def main():
    print("Start of the program")
    workEnglishTask()
    workSpanishTask()
    print("end of the program")

def loadModel(language, task):
    print("training the model for task", task, "in", language)
    if language == "en" :
        file = TRAINING_EN_DATA
    else:
        file = TRAINING_ES_DATA    
    train_data = loadData(file, task, 'train')
    
    #Repeat the process for Test data
    if language == "en" :
        file = TEST_EN_DATA
    else:
        file = TEST_ES_DATA        
    test_data = loadData(file, task, 'test')
    
    return train_data, test_data

def buildFeatureVector(taskData, language):   
    wordsAndClassByTweet = getWordsAndClassByTweet(taskData)
    return getFeatureVector(wordsAndClassByTweet, language)
    

def removeStopWords(allWords, language):
    if language == "en" :
        stopWords = set(stopwords.words('english'))
    else:
        stopWords = set(stopwords.words('spanish'))
    for stopWord in stopWords:
        if(stopWord in allWords):
            allWords.remove(stopWord)
    return allWords
            
def getFeatureNames(wordsAndClassByTweet, language):
    allWords = []
    for sample in wordsAndClassByTweet:
        allWords = allWords + sample[0]
    
    allWords = removeStopWords(allWords, language)
    #Take the 1500 most frequent words and make them as features
    wordBags = nltk.FreqDist(allWords)
    wordBags =  wordBags.most_common(1500)
    return [i[0] for i in wordBags]
 
    
def getFeatureVector(wordsAndClassByTweet, language):
    featureNames = getFeatureNames(wordsAndClassByTweet, language)
    f_vector = []
    for sample in wordsAndClassByTweet:
        tweetWords = sample[0]
        tweetFeatures = {}
        for f_name in featureNames:
            tweetFeatures[f_name] = (f_name in tweetWords)
        f_vector.append([tweetFeatures, sample[1]])
    return f_vector 
    
def loadData(file, task, region):
    data = pd.read_csv(file, delimiter='\t')
    if region == 'train' and task != 'HS' :
        data = data.drop(data[data.HS != 1].index)
    task_data = data[["text", task]]
    task_data.set_index("text", inplace=True)
    return task_data
 
def getWordsAndClassByTweet(taskData):
    wordsAndClassByTweet = []
    for tweetAndClass in taskData.itertuples():
        smallCaseWords = []
        tweet_words = word_tokenize(tweetAndClass[0])
        for tweet_word in tweet_words:
            smallCaseWords.append(tweet_word.lower())
        wordsAndClassByTweet.append([smallCaseWords, tweetAndClass[1]])
    return wordsAndClassByTweet

def NBClassify(train_data, test_data, language):
    training_vector = buildFeatureVector(train_data, language)
    test_vector = buildFeatureVector(test_data, language)
    classifier = nltk.NaiveBayesClassifier.train(training_vector)
    accuracy = nltk.classify.accuracy(classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    classes = classifier.classify_many(test_vector)
    test_tweets = test_data.index.values
    return accuracy, dict(zip(test_tweets, classes))

def SGDClassify(train_data, test_data, language):
    training_vector = buildFeatureVector(train_data, language)
    test_vector = buildFeatureVector(test_data, language)
    SGDClassifier_classifier.train(training_vector)
    accuracy = nltk.classify.accuracy(SGDClassifier_classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    classes = SGDClassifier_classifier.classify_many(test_vector)
    test_tweets = test_data.index.values
    return accuracy, dict(zip(test_tweets, classes))

def LinearSVCClassify(train_data, test_data, language):
    training_vector = buildFeatureVector(train_data, language)
    test_vector = buildFeatureVector(test_data, language)
    SVCClassifier_classifier.train(training_vector)
    accuracy = nltk.classify.accuracy(SVCClassifier_classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    classes = SVCClassifier_classifier.classify_many(test_vector)
    test_tweets = test_data.index.values
    return accuracy, dict(zip(test_tweets, classes))

def NuSVCClassify(train_data, test_data, language):
    training_vector = buildFeatureVector(train_data, language)
    test_vector = buildFeatureVector(test_data, language)
    NuSVC_classifier.train(training_vector)
    accuracy = nltk.classify.accuracy(NuSVC_classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    classes = NuSVC_classifier.classify_many(test_vector)
    test_tweets = test_data.index.values
    return accuracy, dict(zip(test_tweets, classes))

def logisticClassify(train_data, test_data, language):
    training_vector = buildFeatureVector(train_data, language)
    test_vector = buildFeatureVector(test_data, language)    
    log_classifier.train(training_vector)
    accuracy = nltk.classify.accuracy(log_classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    classes = log_classifier.classify_many(test_vector)
    test_tweets = test_data.index.values
    return accuracy, dict(zip(test_tweets, classes))

def voteClassify(train_data, test_data, language):
    test_vector = buildFeatureVector(test_data, language)
    vote_classifier = Voter(log_classifier, SVCClassifier_classifier, NuSVC_classifier)
    accuracy = nltk.classify.accuracy(vote_classifier, test_vector)
    #classifier.show_most_informative_features()
    test_vector = list(zip(*test_vector))[0]
    test_tweets = test_data.index.values
    classes = []
    return accuracy, dict(zip(test_tweets, classes))
    
    
def workEnglishTask():
    print("Working on English task")
    hsPredictions = workTaskA('en')
    workTaskB('en', hsPredictions)

def workSpanishTask():
    print("Working on Spanish task")
    hsPredictions = workTaskA('es')
    #workTaskB('es', hsPredictions)
    
def workTaskA(language):
    print("Working on task A for tweets in", language)
    train_data, test_data = loadModel(language, 'HS')
    accuracy, predictions = NBClassify(train_data, test_data, language)
    print("The NLTK Naive Bayes accuracy for the task A - HS - in language", language, "is", accuracy)

    accuracy, predictions = SGDClassify(train_data, test_data, language)
    print("The SKlearn SGD accuracy for the task A - HS - in language", language, "is", accuracy)
 
    accuracy, predictions = LinearSVCClassify(train_data, test_data, language)
    print("The SKlearn Linear SVC accuracy for the task A - HS - in language", language, "is", accuracy)
   
    accuracy, predictions = NuSVCClassify(train_data, test_data, language)
    print("The SKlearn Nu SVC accuracy for the task A - HS - in language", language, "is", accuracy)

    accuracy, predictions = logisticClassify(train_data, test_data, language)
    print("The SKlearn Logistic Regression accuracy for the task A - HS - in language", language, "is", accuracy)
          
    accuracy, predictions = voteClassify(train_data, test_data, language)
    print("The overall vote based accuracy for the task A - HS - in language", language, "is", accuracy)

    return predictions

def workTaskB(language, hsPredictions):
    print("Working on task B for tweets in", language)
    workSubTaskB(language, hsPredictions, 'TR')
    workSubTaskB(language, hsPredictions, 'AG')
 
def workSubTaskB(language, hsPredictions, task):
    train_data, test_data = loadModel(language, task)
   
    filtered_test_data = test_data
    for k, v in hsPredictions.items():
        if v != 1:
            filtered_test_data.drop(k, inplace=True)
            
    accuracy, tempPredictions = NBClassify(train_data, filtered_test_data, language)
    #print("The accuracy for the task B -", task,"- in language", language, "is", accuracy)
    predictions = {}
    test_tweets = test_data.index.values
    for test_tweet in test_tweets:
        predictions[test_tweet] = tempPredictions.get(test_tweet, 0)
    
    correct_predictions = 0
    for actualTweetAndClass in test_data.itertuples():
        if actualTweetAndClass[1] == predictions[actualTweetAndClass[0]]:
            correct_predictions = correct_predictions + 1
    
    print("The final accuracy for the task B -", task,"- in language", language, "is", correct_predictions / len(predictions))
    return predictions
     
main()