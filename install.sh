# exit when any command fails
set -e
# install prereqs
python3 -m pip install --user --upgrade setuptools wheel virtualenv
export PATH=~/.local/bin:$PATH
# create virtual env
virtualenv release_env -p python3
source release_env/bin/activate
# install hate-detector
pip3 install .
python3 -c "import nltk; nltk.download('stopwords')"
deactivate
