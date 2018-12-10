# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:23:37 2018

@author: vsheth
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd #for processing *.tsv files

#Get the English stop words
enStopWords = set(stopwords.words('english'))

def preProcessTweets(tweets, hs_data):
    #Remove stop words
    for tweet in tweets:
        clean_tweet_words = []
        tweet_words = word_tokenize(tweet)
        for tweet_word  in tweet_words:
            if tweet_word not in enStopWords:
                clean_tweet_words.append(tweet_word.lower()) #convert the tweets to lowercase
                all_words.append(tweet_word.lower()) 
        clean_tweets.append([clean_tweet_words, hs_data.loc[tweet, 'HS']])


def extractCharNGrams(tweets, hs_data, gram):
    global all_words
    for tweet in tweets:
        length = len(tweet)
        tweet_words = [tweet[i:i+gram] for i in range(length - gram + 1)]
        #print(tweet_words)          
        all_words = all_words + tweet_words
        clean_tweets.append([tweet_words, hs_data.loc[tweet, 'HS']])
    print ("done extracting ngrams")
        
def get_features(tweetAndClass):
    tweetFeatures = {}
    for name in featureNames:
        tweetFeatures[name] = (name in tweetAndClass[0])
    return tweetFeatures

def buildFeatureVector():
    featureVector = []
    for tweetAndClass in clean_tweets:
        features = get_features(tweetAndClass)
        featureVector.append([features, tweetAndClass[1]])
    return featureVector
 
 
#Read training dataset
training_data = pd.read_csv(r'C:\Users\viral\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_en_Task A\public_development_en_train_en.tsv', delimiter='\t')
train_tweets = training_data['text']
hs_train_data = training_data[["text", "HS"]]
hs_train_data.set_index("text", inplace=True)

clean_tweets = [] #List of all tweets broken into words minus stopwords
all_words = [] #List of all words in the tweets minus stopwords

#preProcessTweets(train_tweets, hs_train_data)
extractCharNGrams(train_tweets, hs_train_data, 4)

#Take the 1000 most frequent words and make them as features
all_words = nltk.FreqDist(all_words)
#featureNames = list(all_words.keys())[:750]
featureNames =  all_words.most_common(5000)
featureNames = [i[0] for i in featureNames]

trainingVector = buildFeatureVector()

classifier = nltk.NaiveBayesClassifier.train(trainingVector)

#Now build the test dataset

#Read test dataset
test_data = pd.read_csv(r'C:\Users\viral\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_en_Task A\public_development_en_dev_en.tsv', delimiter='\t')
test_tweets = test_data['text']
hs_test_data = test_data[["text", "HS"]]
hs_test_data.set_index("text", inplace=True)

clean_tweets = [] #List of all tweets broken into words minus stopwords
all_words = [] #List of all words in the tweets minus stopwords

#preProcessTweets(test_tweets, hs_test_data)
extractCharNGrams(test_tweets, hs_test_data, 4)

testVector = buildFeatureVector()

accuracy = nltk.classify.accuracy(classifier, testVector)

print("The accuracy is ", accuracy)
