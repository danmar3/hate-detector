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
TODO: Reference Viral's work and describe method with references to figures

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
