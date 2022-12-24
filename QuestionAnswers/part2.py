from pathlib import Path

import pandas as pd
import numpy as np

import DecisionTree as dtree


def q2a():
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


def q2b():
    x_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',]
    f = dtree.calc_entropy
    for f in [dtree.calc_entropy, dtree.calc_gini_index, dtree.calc_majority_error]:
        print('\n', str(f))
        training_acc = pd.Series([])
        test_acc = pd.Series([])
        for i in range(1, len(x_cols) + 1):
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
    
            model = dtree.DecisionTreeModel(X, y, f, max_tree_depth=i)
            verification_accuracy = model.test(X, y) # In this case, should be 100%

            training_acc = training_acc.append(
                pd.Series({i: verification_accuracy})
            )

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

            test_acc = test_acc.append(
                pd.Series({i: verification_accuracy})
            )
        print('\nTRAINING RESULTS')
        print(training_acc)
        print('Results Average: ', training_acc.mean())
        
        print('\nTEST RESULTS')
        print(test_acc)
        print('Results Average: ', test_acc.mean())

    # <function calc_entropy at 0x015B4268>

    # TRAINING RESULTS
    # 1    0.698
    # 2    0.698
    # 3    0.698
    # 4    0.811
    # 5    0.946
    # 6    1.000
    # dtype: float64
    # Results Average:  0.8085

    # TEST RESULTS
    # 1    0.703297
    # 2    0.703297
    # 3    0.703297
    # 4    0.809066
    # 5    0.877747
    # 6    0.875000
    # dtype: float64
    # Results Average:  0.7786172161172161

    # <function calc_gini_index at 0x155F2850>

    # TRAINING RESULTS
    # 1    0.698
    # 2    0.698
    # 3    0.698
    # 4    0.803
    # 5    0.946
    # 6    1.000
    # dtype: float64
    # Results Average:  0.8071666666666667

    # TEST RESULTS
    # 1    0.703297
    # 2    0.703297
    # 3    0.703297
    # 4    0.806319
    # 5    0.877747
    # 6    0.875000
    # dtype: float64
    # Results Average:  0.7781593406593407

    # <function calc_majority_error at 0x155F2898>

    # TRAINING RESULTS
    # 1    0.698
    # 2    0.698
    # 3    0.698
    # 4    0.790
    # 5    0.942
    # 6    1.000
    # dtype: float64
    # Results Average:  0.8043333333333332

    # TEST RESULTS
    # 1    0.703297
    # 2    0.703297
    # 3    0.703297
    # 4    0.791209
    # 5    0.846154
    # 6    0.840659
    # dtype: float64
    # Results Average:  0.7646520146520146


def q3a():
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

    for f in (dtree.calc_entropy, dtree.calc_gini_index, dtree.calc_majority_error):
        print(f'\nError Function------------------{f}')
        training_acc = pd.Series([])
        test_acc = pd.Series([])
        for i in range(1, len(x_cols) + 1):
            print(i)
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
            model = dtree.DecisionTreeModel(X, y, f, 
                max_tree_depth=i, default_value_selection='subset_majority')
            verification_accuracy = model.test(X, y) # In this case, should be 100%

            training_acc = training_acc.append(
                pd.Series({i: verification_accuracy})
            )

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

            test_acc = test_acc.append(
                pd.Series({i: verification_accuracy})
            )

        print('\nTRAINING RESULTS')
        print(training_acc)
        print('Results Average: ', training_acc.mean())

        print('\nTEST RESULTS')
        print(test_acc)
        print('Results Average: ', test_acc.mean())

    # Error Function------------------<function calc_entropy at 0x017F4268>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8808
    # 3     0.8950
    # 4     0.9138
    # 5     0.9370
    # 6     0.9500
    # 7     0.9640
    # 8     0.9698
    # 9     0.9760
    # 10    0.9814
    # 11    0.9852
    # 12    0.9862
    # 13    0.9864
    # 14    0.9864
    # 15    0.9864
    # 16    0.9864
    # dtype: float64
    # Results Average:  0.9541
    
    # TEST RESULTS
    # 1     0.8752
    # 2     0.8752
    # 3     0.8866
    # 4     0.8742
    # 5     0.8628
    # 6     0.8514
    # 7     0.8366
    # 8     0.8312
    # 9     0.8266
    # 10    0.8224
    # 11    0.8154
    # 12    0.8130
    # 13    0.8132
    # 14    0.8098
    # 15    0.8098
    # 16    0.8098
    # dtype: float64
    # Results Average:  0.838325

    # Error Function------------------<function calc_gini_index at 0x15832850>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8912
    # 3     0.9006
    # 4     0.9202
    # 5     0.9386
    # 6     0.9516
    # 7     0.9650
    # 8     0.9714
    # 9     0.9784
    # 10    0.9820
    # 11    0.9848
    # 12    0.9862
    # 13    0.9864
    # 14    0.9864
    # 15    0.9864
    # 16    0.9864
    # dtype: float64
    # Results Average:  0.9560250000000001

    # TEST RESULTS
    # 1     0.8752
    # 2     0.8834
    # 3     0.8856
    # 4     0.8712
    # 5     0.8552
    # 6     0.8424
    # 7     0.8270
    # 8     0.8196
    # 9     0.8136
    # 10    0.8100
    # 11    0.8062
    # 12    0.8018
    # 13    0.8018
    # 14    0.7982
    # 15    0.7982
    # 16    0.7982
    # dtype: float64
    # Results Average:  0.830475

    # Error Function------------------<function calc_majority_error at 0x15832898>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8912
    # 3     0.8988
    # 4     0.9134
    # 5     0.9304
    # 6     0.9422
    # 7     0.9508
    # 8     0.9608
    # 9     0.9672
    # 10    0.9726
    # 11    0.9784
    # 12    0.9826
    # 13    0.9840
    # 14    0.9862
    # 15    0.9864
    # 16    0.9864
    # dtype: float64
    # Results Average:  0.9507625

    # TEST RESULTS
    # 1     0.8752
    # 2     0.8834
    # 3     0.8870
    # 4     0.8758
    # 5     0.8586
    # 6     0.8482
    # 7     0.8342
    # 8     0.8190
    # 9     0.8100
    # 10    0.8038
    # 11    0.7960
    # 12    0.7910
    # 13    0.7872
    # 14    0.7838
    # 15    0.7842
    # 16    0.7842
    # dtype: float64
    # Results Average:  0.8263499999999999


def q3b():
    cols = [
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
    ]

    bank_data = pd.read_csv(
        Path('bank_data', 'train.csv'), 
        names=cols, 
        index_col=False,
    )

    test_bank_data = pd.read_csv(
        Path('bank_data', 'test.csv'), 
        names=cols, 
        index_col=False,
    )

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

    for x_col in x_cols:
        is_unknown = bank_data[x_col] == 'unknown'
        bank_data[x_col].loc[is_unknown] = np.nan
        bank_data[x_col] = bank_data[x_col].fillna(
            bank_data[x_col].loc[~is_unknown].mode()[0]
        )

        is_unknown = test_bank_data[x_col] == 'unknown'
        test_bank_data[x_col].loc[is_unknown] = np.nan
        test_bank_data[x_col] = test_bank_data[x_col].fillna(
            test_bank_data[x_col].loc[~is_unknown].mode()[0]
        )

    for f in (dtree.calc_entropy, dtree.calc_gini_index, dtree.calc_majority_error):
        print(f'Error Function------------------{f}')
        training_acc = pd.Series([])
        test_acc = pd.Series([])
        for i in range(1, len(x_cols) + 1):
            print(i)
            X = bank_data[x_cols]
            y = bank_data.y
            numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            X[numeric_cols] = X[numeric_cols].astype(int)
            model = dtree.DecisionTreeModel(X, y, f, 
                max_tree_depth=i, default_value_selection='subset_majority')
            verification_accuracy = model.test(X, y) # In this case, should be 100%

            training_acc = training_acc.append(
                pd.Series({i: verification_accuracy})
            )

            X = test_bank_data[x_cols]
            y = test_bank_data.y
            numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
            X[numeric_cols] = X[numeric_cols].astype(int)
            verification_accuracy = model.test(X, y) # In this case, should be 100%

            test_acc = test_acc.append(
                pd.Series({i: verification_accuracy})
            )

        print('\nTRAINING RESULTS')
        print(training_acc)
        print('Results Average: ', training_acc.mean())

        print('\nTEST RESULTS')
        print(test_acc)
        print('Results Average: ', test_acc.mean())

    # Error Function------------------<function calc_entropy at 0x00B74268>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8808
    # 3     0.8948
    # 4     0.9104
    # 5     0.9262
    # 6     0.9396
    # 7     0.9528
    # 8     0.9602
    # 9     0.9674
    # 10    0.9726
    # 11    0.9760
    # 12    0.9776
    # 13    0.9780
    # 14    0.9780
    # 15    0.9780
    # 16    0.9780
    # dtype: float64
    # Results Average:  0.94695

    # TEST RESULTS
    # 1     0.8752
    # 2     0.8752
    # 3     0.8860
    # 4     0.8760
    # 5     0.8674
    # 6     0.8578
    # 7     0.8460
    # 8     0.8410
    # 9     0.8352
    # 10    0.8292
    # 11    0.8246
    # 12    0.8214
    # 13    0.8214
    # 14    0.8160
    # 15    0.8160
    # 16    0.8160
    # dtype: float64
    # Results Average:  0.844025
    # Error Function------------------<function calc_gini_index at 0x15772850>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8912
    # 3     0.8964
    # 4     0.9090
    # 5     0.9238
    # 6     0.9388
    # 7     0.9532
    # 8     0.9614
    # 9     0.9692
    # 10    0.9748
    # 11    0.9772
    # 12    0.9780
    # 13    0.9780
    # 14    0.9780
    # 15    0.9780
    # 16    0.9780
    # dtype: float64
    # Results Average:  0.9478625

    # TEST RESULTS
    # 1     0.8752
    # 2     0.8834
    # 3     0.8868
    # 4     0.8766
    # 5     0.8646
    # 6     0.8566
    # 7     0.8402
    # 8     0.8340
    # 9     0.8270
    # 10    0.8214
    # 11    0.8170
    # 12    0.8130
    # 13    0.8132
    # 14    0.8066
    # 15    0.8066
    # 16    0.8066
    # dtype: float64
    # Results Average:  0.8392999999999999
    # Error Function------------------<function calc_majority_error at 0x15772898>

    # TRAINING RESULTS
    # 1     0.8808
    # 2     0.8912
    # 3     0.8978
    # 4     0.9104
    # 5     0.9218
    # 6     0.9304
    # 7     0.9412
    # 8     0.9496
    # 9     0.9572
    # 10    0.9640
    # 11    0.9680
    # 12    0.9726
    # 13    0.9750
    # 14    0.9780
    # 15    0.9780
    # 16    0.9780
    # dtype: float64
    # Results Average:  0.943375

    # TEST RESULTS
    # 1     0.8752
    # 2     0.8834
    # 3     0.8860
    # 4     0.8776
    # 5     0.8716
    # 6     0.8614
    # 7     0.8432
    # 8     0.8294
    # 9     0.8194
    # 10    0.8140
    # 11    0.8080
    # 12    0.8044
    # 13    0.7984
    # 14    0.7910
    # 15    0.7910
    # 16    0.7910
    # dtype: float64
    # Results Average:  0.8340624999999999
