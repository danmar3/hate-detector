"""
@author: Paul Hudgins (hudginspj@vcu.edu)
"""

import gensim
import os
import zipfile
import pandas
import numpy
import random

import embeddings


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


# def load_model(path):
#     model = gensim.models.KeyedVectors.load(path)
#     return model

# def vectorize(word, model):
#     try:
#         return model.word_vec(word)
#     except KeyError:
#         return [0.0] * model.vector_size


def test():
    model = embeddings.load_model("models/wor2vec_filtered_200k.model")
    #model = embeddings.load_model("models/wor2vec_raw_200k.model")
    #model = embeddings.load_model("models/fasttext_filtered_200k.model")

    #print(embeddings.vectorize("#immigrant", model))
    print(model.most_similar("immigrant"))
    print(model.most_similar("woman"))
    print(model)

    

if __name__ == "__main__":
    test()