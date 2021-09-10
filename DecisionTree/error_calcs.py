import math
import numpy as np


def calc_entropy(set_array: np.array):
    h = list()
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        elif p == 0:
            h.append(0)
        else:
            h.append(-p * log2(p))
    return sum(h)


def calc_gini_index(set_array: np.array):
    h = list()
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        elif p == 0:
            h.append(0)
        else:
            h.append(-p**2)
    return 1 + sum(h)


def calc_majority_error(set_array: np.array):
    # print(set_array)
    return min(set_array)


def calc_gain(x: np.array, y: np.array, f=calc_entropy):
    H_y = f(calc_bool_probability(y))
    s = len(y)
    e = list()
    for v in np.unique(x):
        is_x_eq_v = np.where(x==v)
        s_v = len(x[is_x_eq_v])
        y_v = y[is_x_eq_v]
        prob_y_v = calc_bool_probability(y_v)
        e.append(s_v / s * f(prob_y_v))
    return H_y - sum(e)


def calc_bool_probability(x: np.array):
    not_x = np.logical_not(x).astype(int)
    return np.array([x.sum() / len(x), not_x.sum() / len(x)])


def log2(x):
    return math.log(x) / math.log(2)
