# hate-detector Project Organization

## Dataset and Models
The dataset files ```dev_en.tsv, dev_es.tsv, tra_en.tsv, and tra_es.tsv``` must be located in ```hate-detector/nlp516/dataset/development/```.
These files are not included in the github repository because they are restricted to project participants.

The appropriate model files must be located in ```hate-detector/nlp516/dataset/models/```.
These are not included in the github repository due to size.

## Development code
Code that was used in initial experiments is included in ```hate-detector/dev_playgound``` and ```hate-detector/notebooks```.

This is included in our repository for the purposes of sharing between the team, and is not meant as a working product.

## Delivery code
Our core code for stage 2 is included in ```hate-detector/nlp516```. This portion is organized as follows:

* corpus_filters/ - Modules for injesting and filtering a large twitter corpus
* data/ - Modules for data injestion and preprocessing
* dataset/ - Designated folder for training data, corpuses, and models
* lstm/ - A Long-term Short-Term Memory classifier
   * lstm_model/ - lstm model for sequence classification
   * character_lstm/ - lstm classifier with one-hot character vectorizer
   * word2vec_lstm/ - lstm classifier with word embedding vectorizer
* embeddings.py - Trains fasttext and word2vec embeddings and provides wrappers for using saved models.
* main.py - Runs all tests and produces output file results.txt
* model.py - Creates a class for combining vectorizors and classifiers
* vectorizer.py - Defines vectorizors including doc2vec, unigram, and unigram presence

* dev_playground/v1 hatEval.py - Initial experiments with Naieve Bayes
* dev-playground/charNgram_hateval.py -
