import gensim
import data
import os
import zipfile
import nlp516
import pandas
print("hello")
dataset = data.PublicTrialRaw()

import gensim.downloader as api

#word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data

#print(word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset\\public_development_en.zip')
archive = zipfile.ZipFile(src, 'r')
with archive.open(r'public_development_en/train_en.tsv') as file:
            dataset = pandas.read_csv(file, sep='\t')


documents = []
for i in range(dataset.en.shape[0]):
 #   print()
    line = dataset.en.iloc[i].text
    
    prep = gensim.utils.simple_preprocess(line)
    documents.append(prep)
    #print(prep)


model = gensim.models.Word2Vec(
        documents,
        size=20,
        window=10,
        min_count=2,
        workers=10)
model.train(documents, total_examples=len(documents), epochs=10)
#print(documents)
print(model.wv.most_similar("women"))
#print(model.wv['women'])



