"""
Setup of hate-speech packet
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
import nltk
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


class DevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        nltk.download('stopwords')
        develop.run(self)


class InstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        nltk.download('stopwords')
        install.run(self)


# for development installation: pip install -e .
#                               pip install -e .[develop]
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/twodlearn_version.tar.gz
setup(name='nlp516',
      version='0.1',
      packages=find_packages(
          exclude=["*test*", "tests"]),
      install_requires=['nltk', 'pandas==0.23.4', 'pathlib', 'tqdm',
                        'scikit-learn', 'scipy',
                        'matplotlib'],
      python_requires='>=3',
      extras_require={
          'develop': ['nose', 'nose-timer', 'jupyter'],
      },
      cmdclass={
        'develop': DevelopCommand,
        'install': InstallCommand,
        },
      author='team 5',
      licence='GPL',
      url='https://github.com/danmar3/hate-detector'
      )
