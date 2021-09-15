import math
import numpy as np


def calc_entropy(set_array: np.array, unique_outcomes=2) -> float:
    h = list()
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        elif p == 0:
            h.append(0)
        else:
            h.append(-p * logn(p, 2))
    return sum(h)


def calc_gini_index(set_array: np.array, unique_outcomes=2) -> float:
    h = list()
    for p in set_array:
        if p > 1:
            raise ValueError(f'Probability was {p}. Probabilities may not exceed 1.')
        else:
            h.append(-p**2)
    return 1 + sum(h)


def calc_majority_error(set_array: np.array, unique_outcomes=2) -> float:
    if len(set_array) == 1:
        return 0
    return sum(np.sort(set_array)[:-1])


def calc_gain(x: pd.Series, y: pd.Series, f=calc_entropy) -> float:
    is_y_nan = y.isna()
    x = x[~is_y_nan]
    y = y[~is_y_nan]

    H_y = f(calc_discrete_probability(y), unique_outcomes=len(np.unique(y)))
    s = len(y)
    e = list()

    x_no_nans = x[~x.isna()]
    for v in np.unique(x_no_nans):
        # TODO: Get fractional counts to work PJW
        is_x_eq_v = np.where(x==v)
        s_v = len(x[is_x_eq_v])
        y_v = y[is_x_eq_v]
        prob_y_v = calc_discrete_probability(y_v)
        e.append(s_v / s * f(prob_y_v, unique_outcomes=len(np.unique(y))))
    return H_y - sum(e)


def calc_discrete_probability(x: np.array) -> float:
    unique_vals = np.unique(x)
    p = np.array([])
    for i in unique_vals:
        num = sum(x==i)
        p = np.append(p, num / len(x))
    return p


def logn(x, n) -> float:
    return math.log(x) / math.log(n)
