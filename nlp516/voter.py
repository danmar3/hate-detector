"""
Split dataset in K fold
@authors: Viral Sheth,
          Daniel L. Marino (marinodl@vcu.edu)
"""

import nlp516.data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
K_FOLDS = 4

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


def run_experiment(train, test, language, task):
    # Run your code here
   
    hsPredictions = workTaskA(language, train, test)
    if task != 'HS':
        hsPredictions = workTaskB(language, task, hsPredictions, train, test)
        
    return calcMetrics(test, hsPredictions)

def calcMetrics(actual, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    tweets = actual['text']
    for tweet in tweets:
        if actual.loc[tweet, 'HS'] == predictions.get(tweet) == 1:
            TP += 1
        if predictions.get(tweet) == 1 and actual.loc[tweet, 'HS'] != predictions.get(tweet):
            FP += 1
        if actual.loc[tweet, 'HS'] == predictions.get(tweet) == 0:
            TN += 1
        if predictions.get(tweet) == 0 and actual.loc[tweet, 'HS'] != predictions.get(tweet):
            FN += 1
        
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    f1 = (2*precision*recall)/(precision + recall)
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
        
def workTaskA(language, train_data, test_data):
    print("Working on task A for tweets in", language)         
    accuracy, predictions = voteClassify(train_data, test_data, language)
    print("The overall vote based accuracy for the task A - HS - in language", language, "is", accuracy)

    return predictions

def workTaskB(language, task, hsPredictions, train_data, test_data):
    print("Working on task B for tweets in", language)
    return workSubTaskB(language, hsPredictions, task, train_data, test_data)
 
def workSubTaskB(language, hsPredictions, task, train_data, test_data):  
    filtered_test_data = test_data
    for k, v in hsPredictions.items():
        if v != 1:
            filtered_test_data.drop(k, inplace=True)
            
    accuracy, tempPredictions = voteClassify(train_data, filtered_test_data, language)
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
    
def buildFeatureVector(taskData, language):   
    wordsAndClassByTweet = getWordsAndClassByTweet(taskData)
    return getFeatureVector(wordsAndClassByTweet, language)

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
 
def getWordsAndClassByTweet(taskData):
    wordsAndClassByTweet = []
    for tweetAndClass in taskData.itertuples():
        smallCaseWords = []
        tweet_words = word_tokenize(tweetAndClass[0])
        for tweet_word in tweet_words:
            smallCaseWords.append(tweet_word.lower())
        wordsAndClassByTweet.append([smallCaseWords, tweetAndClass[1]])
    return wordsAndClassByTweet


def removeStopWords(allWords, language):
    if language == "en" :
        stopWords = set(stopwords.words('english'))
    else:
        stopWords = set(stopwords.words('spanish'))
    for stopWord in stopWords:
        if(stopWord in allWords):
            allWords.remove(stopWord)
    return allWords