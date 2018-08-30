import os
import nlp516
import zipfile
import pandas as pd


class PublicTrial(object):
    src = os.path.join(os.path.dirname(nlp516.__file__),
                       'dataset/public_trial.zip')

    def __init__(self, src=None):
        archive = zipfile.ZipFile(self.src, 'r')
        with archive.open('trial_en.tsv') as file:
            self.en = pd.read_csv(file, sep='\t')
        with archive.open('trial_es.tsv') as file:
            self.es = pd.read_csv(file, sep='\t')
