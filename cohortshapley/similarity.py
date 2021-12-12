import math
import numpy as np
import pandas as pd

# Similarity Function Example
## unity
def similar_in_unity(data, subject, vertex):
    subject = subject.reshape(subject.shape[-1])
    dataT = data.T
    ccond = np.ones(data.shape[0])
    for i in range(vertex.shape[-1]):
        if vertex[i] == 0:
            continue
        cond = np.equal(dataT[i], subject[i])
        ccond = np.logical_and(ccond, cond)
    return ccond

## distance less than ratio of variable range
ratio = 0.1

def similar_in_distance(data, subject, vertex, categorical=None):
    subject = subject.reshape(subject.shape[-1])
    xmax = np.amax(data,0)
    xmin = np.amin(data,0)
    xdist = (xmax - xmin) * ratio
    cmin = subject - xdist
    cmax = subject + xdist
    dataT = data.T
    ccond = np.ones(data.shape[0])
    for i in range(vertex.shape[-1]):
        if vertex[i] == 0:
            continue
        if (categorical is not None) and (categorical[i] == 1):
            cond = np.equal(dataT[i], subject[i])
        else:
            cond = np.logical_and(np.greater_equal(dataT[i], cmin[i]), np.less_equal(dataT[i], cmax[i]))
        ccond = np.logical_and(ccond, cond)
    return ccond

def set_ratio(x):
    global ratio
    ratio = x


## distance less than ratio of variable range in percentiles
ratio = 0.1
cutoff = 5
def similar_in_distance_cutoff(data, subject, vertex):
    subject = subject.reshape(subject.shape[-1])
    xmax = np.percentile(data,100-cutoff,0)
    xmin = np.percentile(data,cutoff,0)
    xdist = (xmax - xmin) * ratio
    cmin = subject - xdist
    cmax = subject + xdist
    dataT = data.T
    ccond = np.ones(data.shape[0])
    for i in range(vertex.shape[-1]):
        if vertex[i] == 0:
            continue
        cond = np.logical_and(np.greater_equal(dataT[i], cmin[i]), np.less_equal(dataT[i], cmax[i]))
        ccond = np.logical_and(ccond, cond)
    return ccond

def set_cutoff(x):
    global cutoff
    cutoff - x


# put into n bins and similar if in the same bin
bins = 10
def similar_in_samebin(data, subject, vertex):
    subject = subject.reshape(subject.shape[-1])
    xmax = np.amax(data,0)
    xmin = np.amin(data,0)
    xbin = (xmax - xmin) / bins
    cmin = np.multiply(np.floor(np.divide(subject, xbin)), xbin)
    cmax = cmin + xbin
    dataT = data.T
    ccond = np.ones(dataT.shape[-1])
    for i in range(vertex.shape[-1]):
        if vertex[i] == 0:
            continue
        cond = np.logical_and(np.greater_equal(dataT[i], cmin[i]), np.less_equal(dataT[i], cmax[i]))
        ccond = np.logical_and(ccond, cond)
    return ccond

def set_bins(x):
    global bins
    bins = x


### binning
def binning(X, bins=10):
    bin_indices = []
    bin_info_bins = []
    bin_info_x = []
    for j in range(X.shape[1]):
        n_bins = bins
        v = X[:,j]
        bin_vals = np.unique(v)
        rep_x = None
        #print(bin_vals)
        if len(bin_vals) < n_bins:
            bin_vals = np.sort(bin_vals)
            n_bins = len(bin_vals)
            rep_x = np.array(bin_vals)
        else:
            bin_vals = np.linspace(v.min(),v.max()+0.001, n_bins+1, endpoint=True)
            rep_x = np.zeros(n_bins)
            for i in range(n_bins):
                rep_x[i] = (bin_vals[i] + bin_vals[i+1])/2
        bin_index = np.digitize(v, bin_vals, right= False)
        bin_indices.append(bin_index)
        bin_info_bins.append(bin_vals)
        bin_info_x.append(rep_x)
    return np.array(bin_indices).T, bin_info_bins, bin_info_x
