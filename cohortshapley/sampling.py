import math
import numpy as np
import pandas as pd

### uniform distribution
def sample_uniform_distribution(X, n_samples=1000):
    n_vars = X.shape[-1]
    unif_data = np.zeros([n_samples,n_vars], dtype=float)
    for j in range(n_vars):
        vals = X.values.T[j]
        unif_data[:,j] = np.random.uniform(vals.min(), vals.max(), n_samples)
    return unif_data


### joint marginal distribution
def sample_joint_marginal_distribution(X, n_samples=1000):
    n_vars = X.shape[-1]
    marginal = np.zeros([n_samples,n_vars], dtype=float)
    for j in range(n_vars):
        vals = X.values.T[j]
        marginal[:,j] = np.random.choice(vals, n_samples)
    return marginal


### empirical distribution
def sample_empirical_distribution(X, n_samples=100):
    indices = np.random.choice(np.array(range(X.shape[0])), n_samples)
    return X[indices]
