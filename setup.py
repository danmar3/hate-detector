"""
Setup of hate-speech packet
@author: Daniel L. Marino (marinodl@vcu.edu)
"""
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

# for development installation: pip install -e .
#                               pip install -e .[develop]
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/twodlearn_version.tar.gz

def main():
    setup(name='nlp516',
          version='0.1',
          packages=find_packages(
              exclude=["*test*", "tests"]),
          package_data={
              'nlp516': ['dataset/*.zip',
                         'dataset/development/*.tsv']
          },
          install_requires=['nltk==3.3', 'pandas==0.23.4', 'pathlib', 'tqdm',
                            'scikit-learn==0.20.0', 'scipy==1.1.0',
                            'matplotlib==3.0.0', 'gensim==3.6.0',
                            'emoji==0.5.1'],
          extras_require={
              'develop': ['nose', 'nose-timer', 'jupyter'],
          },
          entry_points={'console_scripts': ['run_stage1=nlp516.main:main']},
          author='team 5',
          licence='GPL',
          url='https://github.com/danmar3/hate-detector'
    )


if __name__ == '__main__':
    main()
