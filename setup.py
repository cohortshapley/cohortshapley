from setuptools import setup, find_packages
import sys

setup(name='cohortshapley',
      packages=[package for package in find_packages()
                if package.startswith('cohortshapley')],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'tqdm',
          'scikit-learn'
      ],
      description='Cohort Shapley',
      author='Masayoshi Mase',
      url='https://github.com/cohortshapley/cohortshapley',
      author_email='masayoshi.mase.mh@hitachi.com',
      license='MIT',
      version='0.1.0')
