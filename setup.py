from setuptools import setup, find_packages
# for development installation: pip install -e .
#                               pip install -e .[develop]
# for distribution: python setup.py sdist #bdist_wheel
#                   pip install dist/twodlearn_version.tar.gz
setup(name='nlp516',
      version='0.1',
      packages=find_packages(
          exclude=["*test*", "tests"]),
      install_requires=['pandas==0.23.4', 'pathlib', 'tqdm'],
      python_requires='>=3',
      extras_require={
          'develop': ['nose', 'nose-timer', 'jupyter'],
      },
      author='team 5',
      licence='GPL',
      url='https://github.com/danmar3/hate-detector'
      )
