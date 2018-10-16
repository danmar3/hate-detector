virtualenv release_env -p python3
source release_env/bin/activate
pip install .
python -c "import nltk; nltk.download('stopwords')"
deactivate
