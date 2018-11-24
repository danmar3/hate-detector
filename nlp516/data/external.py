import os
from . import data


def FakeNewsDataset(object):
    src = os.path.join(data.DATA_FOLDER, 'public_trial.zip')

    def __init__(self, src=None):
        archive = zipfile.ZipFile(self.src, 'r')
        with archive.open('trial_en.tsv') as file:
            self.en = pd.read_csv(file, sep='\t')
        with archive.open('trial_es.tsv') as file:
            self.es = pd.read_csv(file, sep='\t')
