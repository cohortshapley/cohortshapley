import math
import numpy as np
import pandas as pd

grand_mean = 0

def ConditionalVariance(y, bin_index, u_set):
    u = np.where(u_set)[0]
    bins = bin_index[:,u]
    bin_dict = {}
    for i in range(len(bins)):
        tup_bin = tuple(bins[i])
        if tup_bin in bin_dict.keys():
            n = bin_dict[tup_bin][0]
            n += 1
            sum = bin_dict[tup_bin][1]
            t = y[i] - grand_mean
            sum += t
            bin_dict[tup_bin] = (n, sum)
        else:
            n = 1
            t = y[i] - grand_mean
            sum = t
            bin_dict[tup_bin] = (n, sum)
    variance = 0
    for key in bin_dict.keys():
        bin = bin_dict[key]
        gain = bin[1] / bin[0]
        variance += gain * gain * bin[0]
    variance /= len(y)
    return variance

def VarianceShapley(y, bin_index):
    global grand_mean
    grand_mean = y.mean()
    n_var = bin_index.shape[-1]
    shapley_values = np.zeros(n_var)
    phi_set = np.zeros(n_var)
    u_k = {}
    u_k[tuple(phi_set)] = 0
    for k in range(n_var):
        coef =  math.factorial(k) * math.factorial(n_var - k - 1) / math.factorial(n_var - 1)
        u_k_base = u_k
        u_k = {}
        for j in range(n_var):
            gain = 0
            for sett in u_k_base.keys():
                set = np.array(sett)
                if set[j] == 1:
                    pass
                else:
                    set_j = set.copy()
                    set_j[j] = 1
                    if tuple(set_j) not in u_k.keys():
                        u_k[tuple(set_j)] = ConditionalVariance(y, bin_index, set_j)
                    gain += u_k[tuple(set_j)] - u_k_base[sett]
            shapley_values[j] += gain * coef / n_var
    return shapley_values
