import pandas as pd

from .error_calcs import calc_gain


def make_decision_tree(X: pd.DataFrame, y: pd.Series, error_f, max_tree_depth=None) -> dict:
    split_node = determine_split(X, y, error_f)
    print(split_node)
    d = {split_node: dict()}
    for v in X[split_node].unique():
        X_v_cols = X.columns[X.columns != split_node]
        X_v = X[X_v_cols].loc[X[split_node] == v]
        y_v = y.loc[X[split_node] == v]
        if len(y_v.unique()) == 1:
            d[split_node][v] = y_v.unique()[0]
        else:
            d[split_node][v] = make_decision_tree(X_v, y_v, error_f)
    return d


def determine_split(X: pd.DataFrame, y: pd.Series, error_f) -> str:
    current_gain = 0
    split_node = X.columns[0]
    for col in X.columns:
        new_gain = calc_gain(X[col].values, y.values, f=error_f)
        if new_gain > current_gain:
            split_node = col + ''
            current_gain = new_gain + 0
    return split_node
