import DecisionTree as dtree
import numpy as np
import pandas as pd

from pathlib import Path


def main():
    car_data = pd.read_csv(
        Path('car_data', 'train.csv'), 
        names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label',], 
        index_col=False,
    )
    X = car_data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',]]
    y = car_data.label
    tree = dtree.make_decision_tree(X, y, error_f=dtree.calc_entropy)

    import json
    print('final tree: ', json.dumps(tree, sort_keys=True, indent=4))

main()