from numpy import mean, std
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode


def grouping(col, data, m):
    unique = data[col].unique()
    pairs = dict()
    for thing in unique:
        means = mean(data[data[col] == thing]['SalePrice'])
        stds = std(data[data[col] == thing]['SalePrice'])
        pairs.update({thing: (means, stds)})
    grouping = dict()
    for key1 in pairs.keys():
        for key2 in pairs.keys():
            if pairs[key1][0] < pairs[key2][0] + m and pairs[key1][0] > pairs[key2][0] - m and key1 != key2:
                data[col] = data[col].replace(to_replace = [key1], value = [key2])
                grouping.update({key1: key2})
        
    return grouping

def assign_grouping(col, data, group):
    for x in data[col]:
        for key, value in group.items():
            if x in value:
                data[col] = data[col].replace(to_replace = [x], value = [key])