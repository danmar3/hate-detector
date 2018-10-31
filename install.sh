# exit when any command fails
set -e
# 1. install prereqs if not installed
command -v virtualenv --version >/dev/null 2>&1 || {
    echo ;
    echo >&2 "virtualenv is not installed. Did you install the pre-requisites? (see README). Aborting...";
    echo "To install pre-requisites, please run:";
    echo "   python3 -m pip install --upgrade setuptools wheel virtualenv ";
    echo "If admin is required, you can use the pip flag --user";
    exit 1;
}
#if ! virtualenv --version; then
#    # python3 -m pip install --user --upgrade setuptools wheel virtualenv
#    exit 1
#fi
# export PATH=~/.local/bin:$PATH
# 2. create virtual env
virtualenv release_env -p python3
source release_env/bin/activate
# 3. install hate-detector
pip3 install .
python3 -c "import nltk; nltk.download('stopwords')"
deactivate
