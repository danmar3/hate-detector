# hate-detector
The hate-detector project contains a classification pipeline for Semeval 2019 task #5.

See ```hate-detector/documentation/Project_Orgaization.md``` for details of project organization.

## Installation (Debian, Ubuntu)
1. Install pre-requisites: Python3, pip, virtualenv:

```
sudo apt-get install python3 python3-pip
python3 -m pip install --upgrade setuptools wheel virtualenv
```

2. Run installation script:
```
chmod +x install.sh
./install.sh
```

## Running tests
Run the test script
```
chmod +x run_tests.sh
./run_tests.sh
```

Results are printed into a set of files called `results_{language}_{task}.txt`,
  where language is either 'english' or 'spanish'.

## Methodology

This project is designed to perform cross-validation on several combinations of classifiers and vectorizations at a single command.

![alt text](https://github.com/danmar3/hate-detector/blob/master/documentation/figures/methodology.jpg?raw=true "Methodology")




### Vectorizations
* Bag-of-Words Vectorizations (BoWV): we
considered the following variations: 1) Presence of unigrams 2) Frequency of unigrams
3) TF-IDF weighted unigrams 3) Presence of
bigrams
* Part-of-Speech (POS) features: we included
POS information following two approaches:
1) Inclusion of POS tag frequencies in addition to BoWV features 2) Combining POS
tags with unigrams before computing vectorizations
* Doc2Vec: Doc2Vec produces document embeddings using a two-layer neural network. It
is an extension of the Word2Vec model developed at Google, and is included in the Python
package Gensim.
* FastText: FastTex is a library for producing
word embeddings, which was developed at
Facebook. It builds as a Python package using Pip, but also requires a C compiler.
* GloVe: Short for Global Vectors, GloVe
develops vector representations of words
based on statistical co-occurrence (Pennington et al., 2014). It was developed at Stanford
and is available as a Python package.
* Character one-hot encoding: this approach
represents the tweet as a sequence of onehot encoded vectors. Each vector represents
a character in the tweet.

### Classifiers

![alt text](https://github.com/danmar3/hate-detector/blob/master/documentation/figures/LSTM_Diagram.png?raw=true "LSTM Structure")

### Corpus Bootstrapping and Building Models
See ```hate-detector/documentation/Building_Models.md``` for instructions to download and filter corpuses and build word embedding models.

### Expected Outputs





## Developers
* Paul Hudgins (hudginspj@.vcu.edu)
  * Stage 1: Experiments with Doc2Vec
  * Stage 2: Procurement of a larger corpus of tweets, comparative evaluation of Word2Vec and FastText, and develpment of corpus bootstrapping method
* Viral Sheth (shethvh@.vcu.edu)
  * Stage  1:  Initial test run using presence of uni-grams as features and NLTK Naive Bayes and SciKitLearn’s Stochastic Gradient Descent, NuSVC as classifiers.
  * Stage2: POS tagging, tf-idf features
* Daniel L. Marino (marinodl@vcu.edu)
  * Stage 1: pre-processing, project integration and project architecture, deployment and final results (Bayes, SVM, logistic regression, random forest, classification trees, gradient boosting, linear SGD)
  * Stage 2: LSTM language models






**Github** repository: https://github.com/danmar3/hate-detector
