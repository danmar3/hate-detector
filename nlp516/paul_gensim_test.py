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
    
    #print(prep)


# model = gensim.models.Word2Vec(
#         documents,
#         size=100,
#         window=10,
#         min_count=2,
#         workers=10)
# model.train(documents, total_examples=len(documents), epochs=10)
# #print(documents)
# print(model.wv.most_similar("refugees"))
# print(len(documents))
# #print(model.wv['women'])


#>>> from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents[:-1000])]
model = Doc2Vec(tagged_docs, vector_size=100, window=2, min_count=1, workers=4)
#vector = model.infer_vector(['kamalaharris', 'illegals', 'dump', 'their', 'kids', 'at', 'the', 'border', 'like', 'road', 'kill', 'and', 'refuse', 'to', 'unite', 'they', 'hope', 'they', 'get', 'amnesty', 'free', 'education', 'and', 'welfare', 'illegal', 'in', 'their', 'country', 'not', 'on', 'the', 'taxpayer', 'dime', 'its', 'scam', 'nodaca', 'noamnesty', 'sendthe'])
#print(vector)
#print(model[2])
#print(tagged_docs[2])


vectors = []
for doc in documents:
    vector = model.infer_vector(doc)
    vectors.append(vector)

#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#classifier = LogisticRegression()
classifier = GradientBoostingClassifier()
classifier.fit(vectors[:-1000], labels[:-1000])

score = classifier.score(vectors[-1000:], labels[-1000:])
print(score)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, vectors, labels, cv=5)

print("Cross validaton accuracy:", numpy.mean(scores), scores)

#for i in range(8000, 9000):
#    print(labels[i], classifier.predict([vectors[i]]), lines[i])

# >>> from gensim.test.utils import get_tmpfile
# >>>
# >>> fname = get_tmpfile("my_doc2vec_model")
# >>>
# >>> model.save(fname)
# >>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!



