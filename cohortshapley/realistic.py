import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import similarity
from . import sampling

### count realistic data
def count_realistic_by_similarity(X_train, X_test, similarity_ratio=0.1, categorical=None):
    n_samples = len(X_test)
    n_features = X_train.shape[-1]
    similarity.ratio = similarity_ratio
    observed = np.zeros(n_samples, dtype=int)
    for i in range(0, n_samples):
        cohort = similarity.similar_in_distance(X_train, X_test[i], np.ones(n_features), categorical)
        n = np.count_nonzero(cohort)
        if n > 0:
            observed[i] = 1
    n_observed = np.count_nonzero(observed)
    return n_observed, n_samples

### calibration of simirality ratio
def calibrate_similarity_ratio(X, Y, ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], num_trials=100, categorical=None, test_size=0.2):
    num_ratios = len(ratios)
    realistic_rates = np.zeros([num_trials, num_ratios])
    for trial in range(num_trials):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=trial)
        for ratio in range(num_ratios):
            count = count_realistic_by_similarity(X_train.values, X_test.values, ratios[ratio], categorical)
            rate = count[0]/count[1]
            realistic_rates[trial, ratio] = rate
    realistic_rates_mean = realistic_rates.mean(axis=0)
    similarity_ratio = -1
    for ratio in range(num_ratios):
        if realistic_rates_mean[ratio] > 0.95:
            similarity_ratio = ratios[ratio]
            break
    return similarity_ratio, realistic_rates


def marginal_similarity_ratio(X, num_samples = 1000, ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], num_trials=10, categorical=None):
    num_ratios = len(ratios)
    realistic_rates = np.zeros([num_trials, num_ratios])
    for trial in range(num_trials):
        X_train = X.values
        X_test = sampling.sample_joint_marginal_distribution(X, num_samples)
        for ratio in range(num_ratios):
            count = count_realistic_by_similarity(X_train, X_test, ratios[ratio], categorical)
            rate = count[0]/count[1]
            realistic_rates[trial, ratio] = rate
    return realistic_rates


import matplotlib.pyplot as plt

def evaluate_similarity_ratio(X,Y,categorical,n_trial=10):
    ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    similarity_ratio1, realistic_rates1 = calibrate_similarity_ratio(X,Y,ratios=ratios,num_trials=n_trial,categorical=categorical,test_size=0.1)
    similarity_ratio2, realistic_rates2 = calibrate_similarity_ratio(X,Y,ratios=ratios,num_trials=n_trial,categorical=categorical,test_size=0.2)
    similarity_ratio3, realistic_rates3 = calibrate_similarity_ratio(X,Y,ratios=ratios,num_trials=n_trial,categorical=categorical,test_size=0.3)
    realistic_rates_marginal = marginal_similarity_ratio(X,num_samples=1000,ratios=ratios,num_trials=n_trial,categorical=categorical)
    return ratios, realistic_rates1, realistic_rates2, realistic_rates3, realistic_rates_marginal




