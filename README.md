
# hate-detector
The hate-detector project contains a classification pipeline for Semeval 2019 task #5.

See ```hate-detector/documentation/Project_Orgaization.md``` for details of project organization.

## Run with one command (not recommended)

The runit.sh script will perform a user install of dependencies, and assumes that python3 is installed.

```
chmod +x runit.sh
./install.sh
```

## Install and run (Debian, Ubuntu)

1. Install pre-requisites: Python3, pip, virtualenv 
```
sudo apt-get install python3 python3-pip
python3 -m pip install --upgrade setuptools wheel virtualenv
```

2. Run installation script: 
```
chmod +x install.sh
./install.sh
```

3. Run the test script
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

We experimented with the following vectorization methods:

* Bag-of-Words Vectorizations (BoWV):
  * Presence of unigrams 
  * Frequency of unigrams
  * TF-IDF weighted unigrams 
  * Presence of bigrams
* Character N-grams
* POS features
* Word2Vec
* FastText
* Character one-hot encoding

### Corpus Bootstrapping and Building Models
Word2Vec and FastText are trained using an external Twitter corpus filtered to resemble the dataset. 
See ```hate-detector/documentation/Building_Models.md``` for instructions to download and filter corpuses and build word embedding models.

### Classifiers
We experimented with the following classifiers:

* Naive Bayes
* Gradient-Boosted Trees
* Linear Regression
* Linear SGD
* Linear Support Vector Classifier
* RBF Support Vector Classifier
* Decision Tree
* A custom LSTM Network (depicted below)

![alt text](https://github.com/danmar3/hate-detector/blob/master/documentation/figures/LSTM_Diagram.png?raw=true "LSTM Structure")



### Expected Outputs





## Developers
* Paul Hudgins (hudginspj@.vcu.edu)
  * Stage 1: Experiments with Doc2Vec
  * Stage 2: Procurement of a larger corpus of tweets, comparative evaluation of Word2Vec and FastText, and develpment of corpus bootstrapping method
* Viral Sheth (shethvh@.vcu.edu)
  * Stage  1:  Initial test run using presence of uni-grams as features and NLTK Naive Bayes and SciKitLearn’s Stochastic Gradient Descent, NuSVC as classifiers.
  * Stage2: POS tagging, character N-gram features
* Daniel L. Marino (marinodl@vcu.edu)
  * Stage 1: pre-processing, project integration and project architecture, deployment and final results (Bayes, SVM, logistic regression, random forest, classification trees, gradient boosting, linear SGD)
  * Stage 2: LSTM language models



**Github** repository: https://github.com/danmar3/hate-detector
