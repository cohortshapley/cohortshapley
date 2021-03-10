import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

# color palette setting
def set_colorblind_palette():
    plt.style.use('seaborn-colorblind')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = [0,4,1,5,2,3]
    plt.rcParams['axes.prop_cycle'] = cycler(color=np.array(colors)[color_cycle].tolist())

def rgb2code(rgb):
    return '#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2])

def set_color_palette(name):
    colors = matplotlib.cm.get_cmap(name).colors
    color_ary = np.floor(np.array(colors)*255).astype(int)
    code = []
    for i in range(len(color_ary)):
        code.append(rgb2code(color_ary[i]))
    plt.rcParams['axes.prop_cycle']  = cycler(color=code)

# aggregated Shapley values
def calc_bottom(bar_l, bar_u, value):
    bottom = np.zeros(len(value))
    for idx in range(len(value)):
        if value[idx] > 0:
            bottom[idx] = bar_u[idx]
            bar_u[idx] += value[idx]
        else:
            bottom[idx] = bar_l[idx]
            bar_l[idx] += value[idx]
    return bar_l, bar_u, bottom

def draw_aggregate_graph(cs_values, columns, predY, base=None, sort2=None,
                  xlabel='Index (ordered by prediction)',
                  ylabel='Impact',
                  ylim=None,
                  order=None):
    if sort2 is not None:
        index = np.argsort(sort2.flatten())
        index2 = np.argsort(predY.flatten()[index], kind='mergesort')
        cs_values_sorted = (cs_values[index])[index2]
    else:
        index = np.argsort(predY.flatten())
        cs_values_sorted = cs_values[index]
    if order is not None:
        column_index=order
    else:
        column_index=range(len(columns))
    plt.grid(b=True, which='major', axis='y')
    bar_l = np.zeros(len(index))
    bar_u = np.zeros(len(index))
    for col in column_index:
        bar_l, bar_u, bottom = calc_bottom(bar_l, bar_u, cs_values_sorted.T[col])
        plt.bar(x=np.array(range(len(index))),height=cs_values_sorted.T[col],label=columns[col],bottom=bottom)
    grand_mean = predY.mean()
    sorted_predY = predY[index]
    if sort2 is not None:
        sorted_predY = sorted_predY[index2]
    plt.plot(sorted_predY - grand_mean, color='blue')
    if base:
        plt.plot(sorted_predY - base, color='black')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])

def draw_aggregate_graph2(cs_values, columns, predY, base=None, sort2=None, multibase=False,
                  xlabel='Index (ordered by prediction)',
                  ylabel='Importance',
                  ylim=None,
                  order=None):
    if sort2 is not None:
        index = np.argsort(sort2.flatten())
        index2 = np.argsort(predY.flatten()[index], kind='mergesort')
        cs_values_sorted = (cs_values[index])[index2]
    else:
        index = np.argsort(predY.flatten())
        cs_values_sorted = cs_values[index]
    if order is not None:
        column_index=order
    else:
        column_index=range(len(columns))
    plt.grid(b=True, which='major', axis='y')
    bar_l = np.zeros(len(index))
    bar_u = np.zeros(len(index))
    for col in column_index:
        bar_l, bar_u, bottom = calc_bottom(bar_l, bar_u, cs_values_sorted.T[col])
        plt.bar(x=np.array(range(len(index))),height=cs_values_sorted.T[col],label=columns[col],bottom=bottom)
    grand_mean = predY.mean()
    sorted_predY = predY[index]
    if sort2 is not None:
        sorted_predY = sorted_predY[index2]
    plt.plot((sorted_predY - grand_mean) * (sorted_predY - grand_mean), color='blue')
    if multibase:
        sorted_predY2 = np.zeros(sorted_predY.shape)
        for i in range(len(index)):
            temp = sorted_predY - sorted_predY[i]
            sorted_predY2 += temp * temp
        sorted_predY2 /= len(index)
        plt.plot(sorted_predY2, color='black')
    if base:
        plt.plot((sorted_predY - base) * (sorted_predY - base), color='black')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
