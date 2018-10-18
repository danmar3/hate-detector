# hate-detector project organization


## Dataset
The dataset files dev_en.tsv, dev_es.tsv, tra_en.tsv, and tra_es.tsv must be located in hate-detector/nlp516/dataset/development/.

These files are not included in the github repository because they are restricted to project participants.

## Development code
Code that was used in initial experiments is included in hate-detector/dev_playgound and hate-detector/notebooks.

This is included in our repository for the purposes of sharing between the team, and is not meant as a working product.


## Delivery code
Our core code for stage 1 is included in hate-detector/nlp516. This portion is organized as follows:

data.py - Reads in the development dataset and performs preprocessing including stemming.
main.py - Runs all tests and produces output file results.txt
model.py - Creates a class for combining vectorizors and classifiers
vectorizer.py - Defines vectorizors including doc2vec, unigram, and unigram presence
