import pandas as pd
import numpy as np

from .error_calcs import calc_gain


class DecisionTreeModel:

    def __init__(self, X: pd.DataFrame, y: pd.Series, error_f, 
            max_tree_depth=None, default_value_selection='majority'):
        self.X = X.copy()
        self.y = y.copy()
        self.default_value_selection = default_value_selection
        self.error_f = error_f
        self.input_max_tree_depth = max_tree_depth
        self.tree = self.make_decision_tree(
            self.convert_numeric_vals_to_categorical(X.copy()), y, 
            error_f=error_f, max_tree_depth=self.max_tree_depth
        )
    
    def convert_numeric_vals_to_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols:
            return X
        for col, m in self.median.iteritems():
            is_gte_m = X[col] >= m
            X[col].loc[is_gte_m] = f'>={m}'
            X[col].loc[~is_gte_m] = f'<{m}'
        return X

    @property
    def numeric_cols(self) -> list:
        return self.X.select_dtypes(include=np.number).columns.tolist()

    @property
    def median(self) -> pd.Series:
        return self.X[self.numeric_cols].median()

    @property
    def max_tree_depth(self):
        if not self.input_max_tree_depth:
            return len(self.X.columns)
        return self.input_max_tree_depth

    @property
    def default_value(self):
        if self.default_value_selection == 'majority':
            return self.y.groupby(self.y).count().idxmax()

    def make_decision_tree(self, X: pd.DataFrame, y: pd.Series, 
            error_f, max_tree_depth=None) -> dict:
        split_node = self.determine_split(X, y, error_f)
        d = {split_node: dict()}
        for v in X[split_node].unique():
            X_v_cols = X.columns[X.columns != split_node]
            X_v = X[X_v_cols].loc[X[split_node] == v]
            y_v = y.loc[X[split_node] == v]
            if len(y_v.unique()) == 1:
                d[split_node][v] = y_v.unique()[0]
            elif max_tree_depth == 1:
                d[split_node][v] = self.default_value
            else:
                d[split_node][v] = self.make_decision_tree(X_v, y_v, 
                    error_f, max_tree_depth - 1)
        return d
    
    @staticmethod
    def determine_split(X: pd.DataFrame, y: pd.Series, error_f) -> str:
        current_gain = 0
        split_node = X.columns[0]
        for col in X.columns:
            new_gain = calc_gain(X[col].values, y.values, f=error_f)
            if new_gain > current_gain:
                split_node = col + ''
                current_gain = new_gain + 0
        return split_node

    def test(self, test_X: pd.DataFrame, test_y: pd.Series) -> float:
        test_X = self.convert_numeric_vals_to_categorical(test_X)
        predict_y = self.evaluate(test_X)
        s = test_y == predict_y
        return s.sum() / s.count()

    def evaluate(self, test_X: pd.DataFrame) -> pd.Series:
        return test_X.apply(self.check_tree, axis=1, args=[self.tree])

    def check_tree(self, row: pd.Series, tree: dict):
        node = list(tree.keys())[0]
        try:
            if isinstance(tree[node][row[node]], str):
                return tree[node][row[node]]
            elif isinstance(tree[node], dict):
                return self.check_tree(row, tree[node][row[node]])
            else:
                return 'TREE FAILED'
        except KeyError:
            return f'TREE FAILED: {node} not in tree.'
