import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

def all_permutations(n_vars):
    return np.array(list(itertools.permutations(range(n_vars))))

class CohortShapley():
    def __init__(self, model, similarity, subject_id, data, func=np.average, y=None,
            parallel=0, pid=0, data_weight=None, permutations=None,
            mc_num = None,
            verbose=1):
        self.model = model
        self.data = data
        self.n_predictors = data.shape[-1]
        self.similarity_function = similarity
        self.subject_id = subject_id
        self.permutations = permutations
        self.func = func

        if  mc_num != None:
            n_vars = data.shape[-1]
            permutations = np.zeros((mc_num, n_vars), dtype=int)
            for k in range(mc_num):
                permutations[k] = np.random.permutation(n_vars)
            self.permutations = permutations
        self.verbose=verbose
        if y is None:
            self.printlog("use given model to predict y.")
            self.y = model(data)
        else:
            self.y = y
            self.printlog("use given y values instead of model prediction.")

        self.grand_mean = self.func(self.y)
        self.parallel = parallel
        self.pid = pid
        self.data_weight = data_weight

    def printlog(self, str):
        if self.verbose>0:
            print(str)

    def save(self, prefix):
        np.save(prefix + '.cs.npy', self.shapley_values)
        np.save(prefix + '.cs2.npy', self.shapley_values2)

    def load(self, prefix):
        self.shapley_values = np.load(prefix + '.cs.npy')
        self.shapley_values2 = np.load(prefix + '.cs2.npy')

    def CohortValue(self, similarity, data, y, subject, vertex):
        cohort = similarity(data, subject, vertex)
        n = np.count_nonzero(cohort)
        if self.data_weight is not None:
            weights = np.extract(cohort, self.data_weight)
            avgv = (weights * np.extract(cohort, y)).sum() / weights.sum()
        else:
            if n == 0:
                avgv = self.func(y)

            else:
                avgv = self.func(np.extract(cohort, y))

        return (n, avgv, cohort)

    def CohortShapleyOne(self, y, similarity, subject_id, data):
        if isinstance(self.permutations, np.ndarray):
            return self.CohortShapleyOnePerm(y, similarity, subject_id, data,
                                        self.permutations)

        n_var = data[0].shape[-1]
        shapley_values = np.zeros(n_var)
        shapley_values2 = np.zeros(n_var)
        phi_set = np.zeros(n_var)
        u_k = {}
        u_k[tuple(phi_set)] = self.CohortValue(similarity, data, y, data[subject_id], phi_set)
        for k in range(n_var):
            coef =  math.factorial(k) * math.factorial(n_var - k - 1) / math.factorial(n_var - 1) / n_var
            u_k_base = u_k
            u_k = {}
            for j in range(n_var):
                gain = 0
                gain2 = 0
                for sett in u_k_base.keys():
                    set = np.array(sett)
                    if set[j] == 1:
                        pass
                    elif u_k_base[sett][0] == 1:
                        pass
                    else:
                        set_j = set.copy()
                        set_j[j] = 1
                        if tuple(set_j) not in u_k.keys():
                            u_k[tuple(set_j)] = self.CohortValue(similarity, data, y,
                                                data[subject_id], set_j)
                        gain_temp = u_k[tuple(set_j)][1] - u_k_base[sett][1]
                        gain += gain_temp
                        t1 = u_k[tuple(set_j)][1] - self.grand_mean
                        t2 = u_k_base[sett][1] - self.grand_mean
                        gain2 += t1 * t1 - t2 * t2
                shapley_values[j] += gain * coef
                shapley_values2[j] += gain2 * coef
        return shapley_values, shapley_values2

    def CohortShapleyOnePerm(self, y, similarity, subject_id, data, permutations):
        n_var = data[0].shape[-1]
        shapley_values = np.zeros(n_var)
        shapley_values2 = np.zeros(n_var)
        phi_set = np.zeros(n_var, dtype=int)
        u_k = {}
        u_k[tuple(phi_set)] = self.CohortValue(similarity, data, y, data[subject_id], phi_set)
        n_perms = len(permutations)
        for p in range(n_perms):
            set_j = phi_set.copy()
            u_base = u_k[tuple(phi_set)]
            for k in range(n_var):
                j = permutations[p,k]
                set_j[j] = 1
                if tuple(set_j) not in u_k.keys():
                    u_j = self.CohortValue(similarity, data, y,
                                           data[subject_id], set_j)
                    u_k[tuple(set_j)] = u_j
                else:
                    u_j = u_k[tuple(set_j)]
                gain = u_j[1] - u_base[1]
                t1 = u_j[1] - self.grand_mean
                t2 = u_base[1] - self.grand_mean
                gain2 = t1 * t1 - t2 * t2
                shapley_values[j] += gain
                shapley_values2[j] += gain2
                u_base = u_j
        shapley_values /= n_perms
        shapley_values2 /= n_perms
        return shapley_values, shapley_values2

    def compute_cohort_shapley(self):
        y = self.y
        similarity = self.similarity_function
        subject_id = self.subject_id
        data = self.data

        if isinstance(self.permutations, np.ndarray):
            self.printlog("compute Shapley values based on permutations")

        if len(subject_id) == 1:
            ret1, ret2 = self.CohortShapleyOne(y, similarity, subject_id[0], data)
            self.shapley_values = np.array([ret1])
            self.shapley_values2 = np.array([ret2])
        elif self.parallel > 0:
            self.printlog("parallel processing with {0} processes".format(self.parallel))
            from multiprocessing import Pool
            idx_list = np.array_split(np.array(subject_id),self.parallel)
            arg_list = []
            for i in range(self.parallel):
                arg_list.append([y, similarity, data, idx_list[i], i,
                                self.data_weight, self.verbose, self.permutations, self.func])
            p = Pool(self.parallel)
            cs_objs = p.map(worker, arg_list)
            p.close()
            shapleys = []
            shapleys2 = []
            for i in range(self.parallel):
                shapleys.append(cs_objs[i].shapley_values)
                shapleys2.append(cs_objs[i].shapley_values2)
            self.shapley_values = np.vstack(shapleys)
            self.shapley_values2 = np.vstack(shapleys2)
        else:
            shapleys = []
            shapleys2 = []
            if self.verbose>0:
                iter = tqdm(subject_id)
            else:
                iter = subject_id
            for i in iter: # tqdm(subject_id):
                ret1, ret2 = self.CohortShapleyOne(y, similarity, i, data)
                shapleys.append(ret1)
                shapleys2.append(ret2)
            self.shapley_values = np.array(shapleys)
            self.shapley_values2 = np.array(shapleys2)

def worker(x):
    y = x[0]
    similarity = x[1]
    data = x[2]
    idx = x[3]
    pid = x[4]
    data_weight = x[5]
    verbose = x[6]
    permutations = x[7]
    func = x[8]
    cs_obj = CohortShapley(None, similarity, idx, data, func=func, y=y, parallel=0, pid=pid,
                           data_weight=data_weight, permutations=permutations, verbose=verbose)
    cs_obj.compute_cohort_shapley()
    return cs_obj
