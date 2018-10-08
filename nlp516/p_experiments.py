import gensim
import data
import os
import zipfile
import nlp516
import pandas
import numpy
print("hello")
dataset = data.PublicTrialRaw()

import gensim.downloader as api

#word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data

#print(word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
#src = os.path.join(os.path.dirname(nlp516.__file__),
#                       'dataset\\public_development_en.zip')
#archive = zipfile.ZipFile(src, 'r')
with open('dataset/public_development_en/train_en.tsv', 'rb') as file:
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
    

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
model = Doc2Vec(tagged_docs, vector_size=300, window=2, min_count=1, workers=4)


vectors = []
for doc in documents:
    vector = model.infer_vector(doc)
    vectors.append(vector)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_validate
scoring = ['precision', 'recall', 'f1', 'accuracy']
# clf = svm.SVC(kernel='linear', C=1, random_state=0)
# scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
#                         cv=5, return_train_score=False)
def clf_eval(classifier):
    scores = cross_validate(classifier, vectors, labels, cv=10, scoring=scoring, return_train_score=False)
    print(type(classifier))
    print("   ", numpy.mean(scores['test_accuracy']), numpy.mean(scores['test_precision']), numpy.mean(scores['test_recall']), numpy.mean(scores['test_f1']))
    # for k, v in scores.items():
    #     print(k, ":", numpy.mean(v))


clf_eval(GradientBoostingClassifier())
clf_eval(RandomForestClassifier())
#clf_eval(SVC())
clf_eval(LogisticRegression())

# scores = cross_val_score(GradientBoostingClassifier(), vectors, labels, cv=5)
# print("GBDT:", numpy.mean(scores), scores)
# # scores = cross_val_score(RandomForestClassifier(), vectors, labels, cv=5)
# # print("RF:", numpy.mean(scores), scores)
# # scores = cross_val_score(SVC(), vectors, labels, cv=5)
# # print("SVM:", numpy.mean(scores), scores)
# scores = cross_val_score(LogisticRegression(), vectors, labels, cv=5)
# print("LR:", numpy.mean(scores), scores)



#for i in range(8000, 9000):
#    print(labels[i], classifier.predict([vectors[i]]), lines[i])

# >>> from gensim.test.utils import get_tmpfile
# >>>
# >>> fname = get_tmpfile("my_doc2vec_model")
# >>>
# >>> model.save(fname)
# >>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!



