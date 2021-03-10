import math
import numpy as np
import pandas as pd
from tqdm import tqdm

class BaselineShapley():
    def __init__(self, model, subject, baseline, subject_id=None,
            eval_realistic=False, similarity=None, data=None):
        self.model = model
        self.subject = subject
        self.baseline = baseline
        self.base_values = model(baseline)
        self.base_value = self.base_values.mean()
        if subject_id is None:
            self.subject_id = np.array(range(subject.shape[0]))
        else:
            self.subject_id = subject_id
        self.n_predictors = baseline.shape[-1]
        self.eval_realistic = eval_realistic
        self.similarity = similarity
        self.data = data
        if self.eval_realistic:
            self.eval_inference_count = 0
            self.eval_realistic_inference_count = 0

    def save(self, prefix):
        np.save(prefix + '.bs.npy', self.shapley_values)
        np.save(prefix + '.bsr.npy', self.shapley_values_realistic)
        np.save(prefix + '.bstc.npy', self.total_count)
        np.save(prefix + '.bsrc.npy', self.realistic_count)
        np.save(prefix + '.bs2.npy', self.shapley_values2)
        np.save(prefix + '.bs2r.npy', self.shapley_values2_realistic)

    def load(self, prefix):
        self.shapley_values = np.load(prefix + '.bs.npy')
        self.shapley_values_realistic = np.load(prefix + '.bsr.npy')
        self.total_count = np.load(prefix + '.bstc.npy')
        self.realistic_count = np.load(prefix + '.bsrc.npy')
        self.shapley_values2 = np.load(prefix + '.bs2.npy')
        self.shapley_values2_realistic = np.load(prefix + '.bs2r.npy')

    def BaselineShapleyCalc(self, model, subject, baseline, vertex, j, data=None):
        if vertex[j] == 1:
            print('error')
            return 0
        # permutation
        input = np.copy(baseline)
        n_vars = subject.shape[-1]
        for i in range(n_vars):
            if vertex[i] > 0:
                input[:,i] = subject[i]
        valv = model(input)
        if self.eval_realistic:
            vertex_all = np.ones(n_vars)
            in_dist = np.ones(baseline.shape[0])
            for i in range(input.shape[0]):
                cohort = self.similarity(data, input[i], vertex_all)
                if np.count_nonzero(cohort) == 0:
                    in_dist[i] = 0
        input[:,j] = subject[j]
        valvj = model(input)
        value = np.mean(valvj) - np.mean(valv)
        t1 = valvj - self.base_values
        t2 = valv - self.base_values
        t3 = (t1 * t1 - t2 * t2)
        value2 = t3.mean()
        if self.eval_realistic:
            for i in range(input.shape[0]):
                cohort = self.similarity(data, input[i], vertex_all)
                if np.count_nonzero(cohort) == 0:
                    in_dist[i] = 0
            total_count = input.shape[0]
            realistic_count = np.count_nonzero(in_dist)
            realistic_value = ((valvj - valv) * in_dist).mean()
            realistic_value2 = (t3 * in_dist).mean()
        else:
            total_count = input.shape[0]
            realistic_count = 0
            realistic_value = 0
            realistic_value2 = 0
        return value, realistic_value, total_count, realistic_count, value2, realistic_value2

    def BaselineShapleyOne(self, model, subject, baseline, subject_id=0, data=None):
        n_var = subject.shape[-1]
        shapley_values = np.zeros(n_var)
        shapley_values_realistic = np.zeros(n_var)
        total_count = 0
        realistic_count = 0
        shapley_values2 = np.zeros(n_var)
        shapley_values2_realistic = np.zeros(n_var)
        for j in range(n_var):
            vertex_list = all_combination(n_var, j)
            for vertex in vertex_list:
                n_players = np.count_nonzero(vertex)
                coef =  math.factorial(n_players) * math.factorial(n_var - n_players - 1) / math.factorial(n_var - 1) / n_var
                ret = self.BaselineShapleyCalc(model, subject, baseline, vertex, j, data)
                temp = ret[0]
                temp_realistic = ret[1]
                total_count += ret[2]
                realistic_count += ret[3]
                shapley_values[j] += temp * coef
                shapley_values_realistic[j] += temp_realistic * coef
                temp2 = ret[4]
                temp2_realistic = ret[5]
                shapley_values2[j] += temp2 * coef
                shapley_values2_realistic[j] += temp2_realistic * coef
        return shapley_values, shapley_values_realistic, total_count, realistic_count, shapley_values2, shapley_values2_realistic

    def compute_baseline_shapley(self):
        model = self.model
        subject = self.subject
        baseline = self.baseline
        subject_id = self.subject_id
        data = self.data
        if len(subject.shape) == 1:
            ret = self.BaselineShapleyOne(model, subject, baseline, subject_id, data)
            self.shapley_values = ret[0]
            self.shapley_values_realistic = ret[1]
            self.total_count = ret[2]
            self.realistic_count = ret[3]
            self.shapley_values2 = ret[4]
            self.shapley_values2_realistic = ret[5]
        else:
            shapley_values = []
            shapley_values_realistic = []
            total_count = []
            realistic_count = []
            shapley_values2 = []
            shapley_values2_realistic = []
            for i in tqdm(subject_id):
                ret = self.BaselineShapleyOne(model, subject[i], baseline, i, data)
                shapley_values.append(ret[0])
                shapley_values_realistic.append(ret[1])
                total_count.append(ret[2])
                realistic_count.append(ret[3])
                shapley_values2.append(ret[4])
                shapley_values2_realistic.append(ret[5])
            self.shapley_values = np.array(shapley_values)
            self.shapley_values_realistic = np.array(shapley_values_realistic)
            self.total_count = np.array(total_count)
            self.realistic_count = np.array(realistic_count)
            self.shapley_values2 = np.array(shapley_values2)
            self.shapley_values2_realistic = np.array(shapley_values2_realistic)

# Tools
def all_combination_players(base_vertex, n, j, players):
    if players == 0:
        return [np.zeros(n)]
    if players == 1:
        vertex_list = []
        for i in range(n):
            if i == j:
                continue
            vertex = np.copy(base_vertex)
            vertex[i] = 1
            vertex_list.append(vertex)
        return vertex_list
    if players == n - 1:
        vertex = np.ones(n)
        vertex[j] = 0
        return [vertex]
    vertex_list = []
    for vertex in base_vertex:
        last_one = np.max(np.where(vertex > 0))
        for i in range(last_one + 1, n):
            if i != j and vertex[i] == 0:
                new_vertex = np.copy(vertex)
                new_vertex[i] = 1
                vertex_list.append(new_vertex)
    return vertex_list

def all_combination(n, j):
    vertex = []
    base_vertex = np.zeros(n)
    vertex.append(base_vertex)
    for players in range(1, n):
        base_vertex = all_combination_players(base_vertex, n, j, players)
        vertex.extend(base_vertex)
    return np.array(vertex)
