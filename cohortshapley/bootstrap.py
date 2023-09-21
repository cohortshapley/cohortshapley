import numpy as np
from tqdm import tqdm
from . import cohortshapley as cs

# Bayesian bootstrap (weighted likelihood bootstrap)
def wlb_cohortshapley(model, similarity, subject_id, data, y=None,
                      parallel=0, pid=0, data_weights=None, bs_size=100,
                      verbose=0):
    n = len(data)
    d = data.shape[-1]
    if data_weights is None:
        data_weights = np.random.exponential(size=n*bs_size).reshape(bs_size,n)
    cs_objs = {}
    if verbose > 0:
        iter = tqdm(range(bs_size))
    else:
        iter = range(bs_size)
    for k in iter:
        cs_objs[k] = cs.CohortShapley(model, similarity, subject_id, data, y=y,
                                  parallel=parallel, data_weight=data_weights[k],
                                  verbose=0)
        cs_objs[k].compute_cohort_shapley()
    bootstrap_shapley_values = np.zeros([bs_size, n, d])
    for k in range(bs_size):
        bootstrap_shapley_values[k] = cs_objs[k].shapley_values
    return bootstrap_shapley_values, data_weights


# path Sampling
def path_mc_cohortshapley(model, similarity, subject_id, data, y=None,
                          parallel=0, func=np.average, permutations=None,
                          mc_num=100, times=100,
                          verbose=0):
    n = len(subject_id)
    d = data.shape[-1]
    if permutations is None:
        permutations = np.zeros((times, mc_num, d), dtype=int)
        for l in range(times):
            for k in range(mc_num):
                permutations[l,k] = np.random.permutation(d)
    cs_objs = {}
    if verbose > 0:
        iter = tqdm(range(times))
    else:
        iter = range(times)
    for k in iter:
        cs_objs[k] = cs.CohortShapley(model, similarity, subject_id, data, func=func, y=y,
                                  parallel=parallel, permutations=permutations[k],
                                  verbose=0)
        cs_objs[k].compute_cohort_shapley()
    sampling_shapley_values = np.zeros([times, n, d])
    sampling_shapley_values2 = np.zeros([times, n, d])
    for k in range(times):
        sampling_shapley_values[k] = cs_objs[k].shapley_values
        sampling_shapley_values2[k] = cs_objs[k].shapley_values2
    return sampling_shapley_values, sampling_shapley_values2, permutations

def union_path_mc_cohortshapley(model, similarity, subject_id, data, union_structure, y=None,
                          parallel=0, func=np.average, permutations=None,
                          mc_num=100, times=100,
                          verbose=0):
    n = len(subject_id)
    d = data.shape[-1]
    m = len(union_structure)
    if permutations is None:
        permutations = np.zeros((times, mc_num, d), dtype=int)
        for l in range(times):
            for k in range(mc_num):
                union_order = np.random.permutation(m)
                perm = []
                for union_ind in union_order:
                    union = union_structure[union_ind]
                    perm = np.append(perm,np.random.permutation(union))
                permutations[l,k] = perm
    cs_objs = {}
    if verbose > 0:
        iter = tqdm(range(times))
    else:
        iter = range(times)
    for k in iter:
        cs_objs[k] = cs.CohortShapley(model, similarity, subject_id, data, func=func, y=y,
                                  parallel=parallel, permutations=permutations[k],
                                  verbose=0)
        cs_objs[k].compute_cohort_shapley()
    sampling_shapley_values = np.zeros([times, n, d])
    sampling_shapley_values2 = np.zeros([times, n, d])
    for k in range(times):
        sampling_shapley_values[k] = cs_objs[k].shapley_values
        sampling_shapley_values2[k] = cs_objs[k].shapley_values2
    return sampling_shapley_values, sampling_shapley_values2, permutations