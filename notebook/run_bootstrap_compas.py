import math
import numpy as np
import pandas as pd

from cohortshapley import dataset
from cohortshapley import similarity
from cohortshapley import cohortshapley as cs
from cohortshapley import bootstrap

import os
import urllib

def compas_recidivism():
    data_dir = 'dataset'
    file_dir = 'dataset/compas'
    file_path = 'dataset/compas/compas-scores-two-years.csv'
    url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(url, file_path)
    df = pd.read_csv(file_path,index_col=False,
                     skipinitialspace=True, na_values='?')
    # preprocess follows Angwin et al. 2016.
    df = df[df['days_b_screening_arrest']<=30]
    df = df[df['days_b_screening_arrest']>=-30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    def age(val):
        res = 0
        if val<25:
            res = 1
        elif val<=45:
            res = 2
        else:
            res = 3
        return res
    def race(val):
        res = 0
        if val == 'Caucasian':
            res = 1
        elif val == 'African-American':
            res = 2
        elif val == 'Hispanic':
            res = 3
        elif val == 'Asian':
            res = 4
        elif val == 'Native American':
            res = 5
        else:
            res = 6
        return res
    # The definition of categories for prior_count is from Chouldecova 2017.
    def prior(val):
        if val == 0:
            res = 0
        elif val <= 3:
            res = 1
        elif val <= 6:
            res = 2
        elif val <= 10:
            res = 3
        else:
            res = 4
        return res
    df['priors_count'] = df['priors_count'].apply(prior)
    df['crime_factor'] = df['c_charge_degree'].apply(lambda x: 1 if x == 'M' else 2)
    df['age_factor'] = df['age'].apply(age)
    df['race_factor'] = df['race'].apply(race)
    df['gender_factor'] = df['sex'].apply(lambda x: 1 if x == 'Male' else 2)
    df['score_factor'] = df['decile_score'].apply(lambda x: 1 if x >= 5 else 0)
    #  select data subjects of White and Black
    df = df[df['race_factor'] <= 2]
    X = df[['priors_count','crime_factor','age_factor','race_factor','gender_factor']]
    Y = df[['score_factor','two_year_recid','decile_score']]
    return X, Y, df

X, Y, df = compas_recidivism()
X_wr = X.copy()
X_wor = X.drop('race_factor', axis=1)
Y['TP'] = ((Y['score_factor'] == 1).values & (Y['two_year_recid'] == 1).values).astype(int)
Y['FP'] = ((Y['score_factor'] == 1).values & (Y['two_year_recid'] == 0).values).astype(int)
Y['TN'] = ((Y['score_factor'] == 0).values & (Y['two_year_recid'] == 0).values).astype(int)
Y['FN'] = ((Y['score_factor'] == 0).values & (Y['two_year_recid'] == 1).values).astype(int)
Y['res'] = Y['two_year_recid'] - Y['score_factor']


bs_size = 1000

### residual
subject = X_wr.values
similarity.ratio = 0.1
f=False
bootstrap_cs_res, data_weights = bootstrap.wlb_cohortshapley(f, similarity.similar_in_samebin,
    np.arange(len(subject)), subject, y=Y['res'].values, parallel=4, bs_size=bs_size, verbose=1)
np.save('bootstrap_cs_res.npy', bootstrap_cs_res)
np.save('bootstrap_cs_data_weights.npy', data_weights)

### ground truth response
subject = X_wr.values
similarity.ratio = 0.1
f=False
bootstrap_cs_truth, data_weights2 = bootstrap.wlb_cohortshapley(f, similarity.similar_in_samebin,
    np.arange(len(subject)), subject, y=Y['two_year_recid'].values, parallel=4, bs_size=bs_size, verbose=1,
    data_weights=data_weights)
np.save('bootstrap_cs_truth.npy', bootstrap_cs_truth)

### binary prediction
subject = X_wr.values
similarity.ratio = 0.1
f=False
bootstrap_cs_pred, data_weights2 = bootstrap.wlb_cohortshapley(f, similarity.similar_in_samebin,
    np.arange(len(subject)), subject, y=Y['score_factor'].values, parallel=4, bs_size=bs_size, verbose=1,
    data_weights=data_weights)
np.save('bootstrap_cs_pred.npy', bootstrap_cs_pred)

### false positive
subject = X_wr.values
similarity.ratio = 0.1
f=False
bootstrap_cs_fp, data_weights2 = bootstrap.wlb_cohortshapley(f, similarity.similar_in_samebin,
    np.arange(len(subject)), subject, y=Y['FP'].values, parallel=4, bs_size=bs_size, verbose=1,
    data_weights=data_weights)
np.save('bootstrap_cs_fp.npy', bootstrap_cs_fp)

### false negative
subject = X_wr.values
similarity.ratio = 0.1
f=False
bootstrap_cs_fn, data_weights2 = bootstrap.wlb_cohortshapley(f, similarity.similar_in_samebin,
    np.arange(len(subject)), subject, y=Y['FN'].values, parallel=4, bs_size=bs_size, verbose=1,
    data_weights=data_weights)
np.save('bootstrap_cs_fn.npy', bootstrap_cs_fn)
