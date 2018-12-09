# hate-detector Project Organization

Models are provided pre-built because corpus filtering and model training takes more than a day to complete.

To duplicate this process, follow the following steps:

## Download tweets 

Download tweets from: https://archive.org/details/archiveteam-twitter-stream-2017-02

These must be saved in the folder ```hate-detector/nlp516/dataset/corpuses/```

Alternately, any body of tweets can be used from: https://archive.org/search.php?query=collection%3Atwitterstream&sort=-publicdate

##Filter the corpus
```cd hate-detector/nlp516/```

Filter based on language: ```python corpus_filters/language_filter.py```

Filter based on similarity to dataset: ```python corpus_filters/language_filter.py```

## Train embedding models
```cd hate-detector/nlp516/```

```python embeddings.py```
