# hate-detector
The hate-detector project contains work in progress towards Semeval 2019 task #5.

Our first stage of this project includes:
- Data preprocessing including stemming and removal of stop words
- Experiments comparing vectorizations
- Experiments comparing classifiers

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
* Viral Sheth (shethvh@.vcu.edu)
* Daniel L. Marino (marinodl@vcu.edu)
