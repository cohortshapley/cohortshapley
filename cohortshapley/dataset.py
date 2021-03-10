import math
import numpy as np
import pandas as pd
import sklearn.datasets
import os
import urllib.request

# X: input variables (Pandas Dataframe)
# Y: output variable (Numpy Array)

def boston_housing():
    d = sklearn.datasets.load_boston()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    X,Y = df, d.target
    categorical = None
    return X,Y,categorical

def titanic():
    # titanic3.csv is obtained from http://hbiostat.org/data
    # courtesy of the Vanderbilt University Department of Biostatistics.
    file_dir = 'dataset/titanic'
    file_path = 'dataset/titanic/titanic3.csv'
    url = 'https://hbiostat.org/data/repo/titanic3.csv'
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(url, file_path)
    df = pd.read_csv(file_path)
    # preprocessing
    df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
    df = df[['pclass','sex','age','sibsp','parch','fare','survived']]
    df = df.dropna()
    X = df[['pclass','sex','age','sibsp','parch','fare']]
    Y = pd.DataFrame(df['survived'], columns=['survived']).values
    categorical = np.array([1, 1, 0, 0, 0, 0], dtype=int)
    return X,Y,categorical
