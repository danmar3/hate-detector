# -*- coding: utf-8 -*-
"""
Created on Mon Dec 8 15:29:47 2018

@author: viral
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd #for processing *.tsv files

#Get the English stop words
esStopWords = set(stopwords.words('spanish'))

def preProcessTweets(tweets, hs_data):
    #Remove stop words
    for tweet in tweets:
        clean_tweet_words = []
        tweet_words = word_tokenize(tweet)
        for tweet_word  in tweet_words:
            if tweet_word not in esStopWords:
                clean_tweet_words.append(tweet_word.lower()) #convert the tweets to lowercase
                all_words.append(tweet_word.lower()) 
        clean_tweets.append([clean_tweet_words, hs_data.loc[tweet, 'HS']])


def extractPOSTags(tweets, hs_data):
    global all_words
    for tweet in tweets:
        wordsList = word_tokenize(tweet) 
  
        # removing stop words from wordList 
        wordsList = [w for w in wordsList if not w in esStopWords]  
  
        tagged = nltk.pos_tag(wordsList)
        tweet_tags = []
        for (word, tag) in tagged:
            tweet_tags.append(tag)
    
        #print(tweet_tags)    
        #all_words.update(tweet_tags)
        all_words.update(set(tweet_tags))
        clean_tweets.append([tweet_tags, hs_data.loc[tweet, 'HS']])
    #print(all_words)
        
def get_features(tweetAndClass):
    tweetFeatures = {}
    for name in featureNames:
        tweetFeatures[name] = tweetAndClass[0].count(name)
    return tweetFeatures

def buildFeatureVector():
    featureVector = []
    for tweetAndClass in clean_tweets:
        features = get_features(tweetAndClass)
        featureVector.append([features, tweetAndClass[1]])
    #print(featureVector)
    return featureVector
 
 
#Read training dataset
training_data = pd.read_csv(r'C:\Users\viral\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_Spanish_Task A\public_development_es_train_es.tsv', delimiter='\t')
train_tweets = training_data['text']
hs_train_data = training_data[["text", "HS"]]
hs_train_data.set_index("text", inplace=True)

clean_tweets = [] #List of all tweets broken into words minus stopwords
all_words = set() #set of all words in the tweets minus stopwords

#preProcessTweets(train_tweets, hs_train_data)
extractPOSTags(train_tweets, hs_train_data)

#print(all_words)

featureNames =  list(all_words)

trainingVector = buildFeatureVector()

classifier = nltk.NaiveBayesClassifier.train(trainingVector)

#Now build the test dataset

#Read test dataset
test_data = pd.read_csv(r'C:\Users\viral\Documents\CMSC 516 Advanced NLP\SemEval\Test data\public_development_Spanish_Task A\public_development_es_dev_es.tsv', delimiter='\t')
test_tweets = test_data['text']
hs_test_data = test_data[["text", "HS"]]
hs_test_data.set_index("text", inplace=True)

clean_tweets = [] #List of all tweets broken into words minus stopwords
all_words = set() #List of all words in the tweets minus stopwords

#preProcessTweets(test_tweets, hs_test_data)
extractPOSTags(test_tweets, hs_test_data)

testVector = buildFeatureVector()
print(len(testVector))

accuracy = nltk.classify.accuracy(classifier, testVector)

print("The accuracy is ", accuracy)
