import math
import numpy as np


def calc_entropy(set_array: np.array, unique_outcomes=2):
    h = list()
    print(set_array)
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        elif p == 0:
            h.append(0)
        else:
            h.append(-p * logn(p, unique_outcomes))
    return sum(h)


def calc_gini_index(set_array: np.array, unique_outcomes=2):
    h = list()
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        elif p == 0:
            h.append(0)
        else:
            h.append(-p**2)
    return 1 + sum(h)


def calc_majority_error(set_array: np.array, unique_outcomes=2):
    return min(set_array)


def calc_gain(x: np.array, y: np.array, f=calc_entropy):
    H_y = f(calc_discrete_probability(y), unique_outcomes=len(np.unique(y)))
    s = len(y)
    e = list()
    for v in np.unique(x):
        is_x_eq_v = np.where(x==v)
        s_v = len(x[is_x_eq_v])
        y_v = y[is_x_eq_v]
        prob_y_v = calc_discrete_probability(y_v)
        e.append(s_v / s * f(prob_y_v, unique_outcomes=len(np.unique(y))))
    return H_y - sum(e)


def calc_discrete_probability(x: np.array):
    unique_vals = np.unique(x)
    p = np.array([])
    for i in unique_vals:
        num = sum(x==i)
        p = np.append(p, num / len(x))
    return p


def logn(x, n):
    return math.log(x) / math.log(n)
