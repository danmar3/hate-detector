# hate-detector
The hate-detector project contains work in progress towards Semeval 2019 task #5.

The first stage of our project includes:

* Data preprocessing including stemming and removal of stop words
* Experiments comparing vectorizations
* Experiments comparing classifiers

See hate-detector/documentation/Project_Orgaization.md for more information.

## Installation (Debian, Ubuntu)
1. Install pre-requisites: Python3, pip, virtualenv:

```
sudo apt-get install python3 python3-pip
sudo pip install virtualenv
```

2. Run installation script:
```
chmod +x install.sh
./install.sh
```

## Running tests
Run the test script
```
chmod +x stage1_tests.sh
./stage1_tests.sh
```

Results are printed into a file called `results.txt`

## Developers
* Paul Hudgins (hudginspj@.vcu.edu)
  * Stage 1: Experiments with Doc2Vec
  * Stage 2: Procurement of a larger corpus of tweets, and comparative evaluation of embedding systems such as Word2Vec, GloVe, and FastText.
* Viral Sheth (shethvh@.vcu.edu)
  * Stage  1:  Initial test run using presence of uni-grams as features and NLTK Naive Bayes and SciKitLearn’s Stochastic Gradient Descent, NuSVC as classifiers.
  * Stage2: POS tagging, tf-idf features
* Daniel L. Marino (marinodl@vcu.edu)
  * Stage 1: pre-processing, project integration and project architecture, deployment and final results (Bayes, SVM, logistic regression, random forest, classification trees, gradient boosting, linear SGD)
  * Stage 2: LSTM language models


