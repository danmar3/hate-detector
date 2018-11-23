import gensim
#import nlp516.data as data
import os
import zipfile
#import nlp516
import pandas
import numpy
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def main():
    model = Doc2Vec.load("test.model")

    classifier = train_cls(model)
    print("done training classifier")

    with open("corpus.txt") as f:
        count = 0
        positives = []
        negatives = []
        for line in f:
            count += 1
            #print(line)
            
            pred = predict_tweet(classifier, model, line)[0]
            if pred == 1:
                positives.append(line)
            else:
                negatives.append(line)

            if count % 10000 == 0:
                print(count)
            if count > 20000:
                break
        # print("--------- negatives  ------------")
        # for l in negatives:
        #     print(l)
        # print("--------- positives ------------")
        # for l in positives:
        #     print(l)
        print("done")
        print(len(positives))
        print(len(negatives))
        print(count)


def predict_tweet(classifier, model, doc):
    vector = model.infer_vector(doc)

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        
    return classifier.predict([vector])

def train_cls(model):
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

    
    vectors = []
    for doc in documents:
        vector = model.infer_vector(doc)
        vectors.append(vector)

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    classifier = RandomForestClassifier()
    classifier.fit(vectors, labels)
    return classifier





if __name__ == "__main__":
    main()