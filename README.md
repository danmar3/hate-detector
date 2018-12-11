
# hate-detector
The hate-detector project contains a classification pipeline for Semeval 2019 task #5.

This classification task identifies Twitter hate speech against immigrants and women,
and further identifies if the hate speech is aggressive or targeted.

See ```hate-detector/documentation/Project_Orgaization.md``` for details of project organization.

## Run all tests with one command (not recommended)

The runit.sh script will perform a user install of dependencies, and assumes that python3 is installed.

```
chmod +x runit.sh;
./install.sh
```

## Install and run (Debian, Ubuntu)

1. Install pre-requisites: Python3, pip, virtualenv
```
sudo apt-get install python3 python3-pip;
python3 -m pip install --upgrade setuptools wheel; virtualenv
```

2. Run installation script:
```
chmod +x install.sh;
./install.sh
```

3. Run selected tests:
```
chmod +x run_english_hs.sh;
./run_english_hs.sh
```

Alternately, run all tests (may take several hours):
```
chmod +x run_tests.sh;
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
* Character N-grams
* POS features
* Word2Vec
* FastText
* Character one-hot encoding

### Corpus Bootstrapping and Building Models
Word2Vec and FastText are trained using an external Twitter corpus filtered to resemble the dataset.
See ```hate-detector/documentation/Building_Models.md``` for instructions to download and filter corpuses and build the word embedding models.

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
The following is an example of one of the output files
(named ```results_english_HS.txt```).
  The file contains the 4-fold validation results.



```
------------------------------english HS------------------------------

accuracy        f1  precision    recall

MajorityBaseline -            0.5790  0.00000 0.000000  0.000000
bayes            frequency    0.7085  0.607984   0.700072  0.537352
                 presence     0.7131  0.627779   0.691339  0.574954
forest           frequency    0.7413  0.677892   0.712427  0.646830
                 presence     0.7440  0.681259   0.715816  0.650189
gradient-boosted frequency    0.7539  0.654647   0.800273  0.554244
                 presence     0.7538  0.655037   0.799004  0.555367
linear           frequency    0.7556  0.673503   0.769547  0.598851
                 presence     0.7555  0.676236   0.764150  0.606513
linear-sgd       frequency    0.7388  0.669984   0.718742  0.630984
                 presence     0.7276  0.634213   0.738802  0.570462
svc-linear       frequency    0.7564  0.669519   0.780466  0.586247
                 presence     0.7552  0.671469   0.771975  0.594209
svc-rbf          frequency    0.7535  0.659596   0.788246  0.567678
                 presence     0.7475  0.654188   0.773200  0.567933
tree             frequency    0.7166  0.650617   0.676402  0.626938
                 presence     0.7197  0.656507   0.678042  0.636475
```

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
