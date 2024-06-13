import pandas as pd
import numpy as np


def sturges_rule(data):
    return int(np.ceil(np.log2(len(data)) + 1))

def rice_rule(data):
    return int(np.ceil(2 * np.sqrt(len(data))))

def scotts_rule(data):
    h = 3.5 * np.std(data) / (len(data) ** (1/3))
    return int(np.ceil((np.max(data) - np.min(data)) / h))

def freedman_diaconis_rule(data):
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    h = 2 * iqr / (len(data) ** (1/3))
    return int(np.ceil((np.max(data) - np.min(data)) / h))

def doanes_rule(data):
    n = len(data)
    g1 = pd.Series(data).skew()
    sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
    return int(1 + np.log2(n) + np.log2(1 + np.abs(g1) / sigma_g1))

def sqrt_choice(data):
    return int(np.ceil(np.sqrt(len(data))))
