import os
import pandas as pd
from . import data


def load_fakenews():
    src = os.path.join(data.DATA_FOLDER, 'external/fake-news/fake.csv')
    with open(src) as file:
        raw = pd.read_csv(file)
    raw = raw[[type(x) is str for x in raw.text]].reindex()
    return raw
