import DecisionTree as dtree
import numpy as np
import pandas as pd

from pathlib import Path

pd.options.mode.chained_assignment = None  # default='warn'


def main():
    for f in [dtree.calc_entropy, dtree.calc_gini_index, dtree.calc_majority_error]:
        print(str(f))
        car_data = pd.read_csv(
            Path('car_data', 'train.csv'), 
            names=[
                'buying', 
                'maint', 
                'doors', 
                'persons', 
                'lug_boot', 
                'safety', 
                'label',
            ], 
            index_col=False,
        )
        X = car_data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',]]
        y = car_data.label

        model = dtree.DecisionTreeModel(X, y, f)
        verification_accuracy = model.test(X, y) # In this case, should be 100%
        print(verification_accuracy)

        test_car_data = pd.read_csv(
            Path('car_data', 'test.csv'), 
            names=[
                'buying', 
                'maint', 
                'doors', 
                'persons', 
                'lug_boot', 
                'safety', 
                'label',
            ],  
            index_col=False,
        )
        X = test_car_data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',]]
        y = test_car_data.label

        verification_accuracy = model.test(X, y) # In this case, should be 100%
        print(verification_accuracy)

        x_cols = [
            'age',
            'job',
            'marital',
            'education',
            'default',
            'balance',
            'housing',
            'loan',
            'contact',
            'day',
            'month',
            'duration',
            'campaign',
            'pdays',
            'previous',
            'poutcome',
        ]

    for i in range(1, len(x_cols) + 1):
        print(f'Tree Depth----------------------------------------{i}')
        print('TRAIN-----------------------------------------------')
        bank_data = pd.read_csv(
            Path('bank_data', 'train.csv'), 
            names=[
                'age',
                'job',
                'marital',
                'education',
                'default',
                'balance',
                'housing',
                'loan',
                'contact',
                'day',
                'month',
                'duration',
                'campaign',
                'pdays',
                'previous',
                'poutcome',
                'y',
            ], 
            index_col=False,
        )
        X = bank_data[x_cols]
        y = bank_data.y
        numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        X[numeric_cols] = X[numeric_cols].astype(int)
        model = dtree.DecisionTreeModel(X, y, dtree.calc_entropy, 
            max_tree_depth=i, default_value_selection='subset_majority')
        verification_accuracy = model.test(X, y) # In this case, should be 100%
        print(verification_accuracy)

        print('TEST------------------------------------------------')
        test_bank_data = pd.read_csv(
            Path('bank_data', 'test.csv'), 
            names=[
                'age',
                'job',
                'marital',
                'education',
                'default',
                'balance',
                'housing',
                'loan',
                'contact',
                'day',
                'month',
                'duration',
                'campaign',
                'pdays',
                'previous',
                'poutcome',
                'y',
            ], 
            index_col=False,
        )
        X = test_bank_data[x_cols]
        y = test_bank_data.y
        numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        X[numeric_cols] = X[numeric_cols].astype(int)
        verification_accuracy = model.test(X, y) # In this case, should be 100%
        print(verification_accuracy)

        # import json
        # print('final tree: ', json.dumps(model.tree, sort_keys=True, indent=2))

main()