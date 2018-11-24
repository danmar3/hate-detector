"""
@author: Paul Hudgins (hudginspj@vcu.edu)
"""

import gensim
import os
import zipfile
import pandas
import numpy
import random

import word2vec_train


# with open('../nlp516/dataset/development/train_en.tsv', 'rb') as f:
#             dataset = pandas.read_csv(f, sep='\t')

# lines = []
# documents = []
# labels = []
# for i in range(dataset.shape[0]):
#  #   print()
#     line = dataset.iloc[i].text
#     lines.append(line)
#     label = dataset.iloc[i].HS
#     labels.append(label)
#     #print(label)
#     prep = gensim.utils.simple_preprocess(line)
#     documents.append(prep)






#########################################################################


# tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]


# model = Doc2Vec.load("test.model")

# vectors = []
# for doc in documents:
#     vector = model.infer_vector(doc)
#     vectors.append(vector)

# z = list(zip(vectors, labels))
# random.shuffle(z)
# vectors, labels = zip(*z)


# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# from sklearn.model_selection import cross_val_score, cross_validate
# scoring = ['precision', 'recall', 'f1', 'accuracy']
# def clf_eval(classifier):
#     scores = cross_validate(classifier, vectors, labels, cv=10, scoring=scoring, return_train_score=False)
#     print(type(classifier))
#     print("   a/p/r/f: ", numpy.mean(scores['test_accuracy']), numpy.mean(scores['test_precision']), numpy.mean(scores['test_recall']), numpy.mean(scores['test_f1']))
#     # for k, v in scores.items():
#     #     print(k, ":", numpy.mean(v))

# print("evaluating")
# #clf_eval(GradientBoostingClassifier())
# clf_eval(RandomForestClassifier())
# #clf_eval(SVC())
# # clf_eval(LogisticRegression())

def test():
    from word2vec_train import vectorize
    model = word2vec_train.load_model()
    print(model.most_similar(positive="refugee"))
    # print(model.wv.most_similar(positive="refugee"))
    # print(model.wv.vector_size)
    #print(model.word_vec("refugee"))
    print(vectorize("is", model))
    
    

if __name__ == "__main__":
    test()