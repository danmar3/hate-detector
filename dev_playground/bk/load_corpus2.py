print("starting")
import gensim
#import nlp516.data as data
import os
import zipfile
#import nlp516
print()
import pandas
import numpy
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def main():
    print("hello")
    pass
    # classifier = train_cls(model)
    # print("done training classifier")

    # with open("corpus.txt") as f:
    #     count = 0
    #     positives = []
    #     negatives = []
    #     for line in f:
    #         count += 1
    #         #print(line)
            
    #         pred = predict_tweet(classifier, model, line)[0]
    #         if pred == 1:
    #             positives.append(line)
    #         else:
    #             negatives.append(line)

    #         if count % 10000 == 0:
    #             print(count)
    #         if count > 20000:
    #             break
    #     # print("--------- negatives  ------------")
    #     # for l in negatives:
    #     #     print(l)
    #     # print("--------- positives ------------")
    #     # for l in positives:
    #     #     print(l)
    #     print("done")
    #     print(len(positives))
    #     print(len(negatives))
    #     print(count)


def predict_tweet(classifier, model, doc):
    vector = model.infer_vector(doc)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        
    return classifier.predict([vector])

def get_train_data():
    with open('nlp516/dataset/development/train_en.tsv', 'rb') as file:
            dataset = pandas.read_csv(file, sep='\t')

    lines = []
    documents = []
    labels = []
    for i in range(dataset.shape[0]):
    #   print()
        line = dataset.iloc[i].text
        lines.append(line)
        label = dataset.iloc[i].HS
        labels.append(label)
        #print(label)
        prep = gensim.utils.simple_preprocess(line)
        documents.append(prep)
    return lines, labels, documents


def prep_tweet(text):
    allWords = word_tokenize(text)
    allWords = [word.lower() for word in allWords]
    stopWords = set(stopwords.words('spanish'))
    for stopWord in stopWords:
        if(stopWord in allWords):
            allWords.remove(stopWord)
    return allWords

def train_cls():
    lines, labels, documents = get_train_data()
    pass
    # tweet_words = [prep_tweet(words) for words in word_tokenize(lines)]
    tweet_words = [prep_tweet(text) for text in lines]
    from sklearn.feature_extraction.text import CountVectorizer
    X = CountVectorizer(max_features=1000).fit_transform(lines).toarray()
    #wordsAndClassByTweet = list(zip(tweet_words, labels))
    classifier = RandomForestClassifier()
    classifier.fit(X, labels)
    

    # for t in wordsAndClassByTweet:
    #     print(t)

    #wordsAndClassByTweet = getWordsAndClassByTweet(taskData)
    #vector = getFeatureVector(wordsAndClassByTweet, language)
    # classifier = nltk.NaiveBayesClassifier.train(training_vector)
    # classifier.classify("test test test")

    # classifier = RandomForestClassifier()
    # classifier.fit(vectors, labels)
    # return classifier




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
    #Take the 1000 most frequent words and make them as features
    wordBags = nltk.FreqDist(allWords)
    wordBags =  wordBags.most_common(750)
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



if __name__ == "__main__":
    #main()
    train_cls()