import numpy as np
import pandas as pd

import DecisionTree as dtree


def print_vals_of_unique_labels(y, x):
    for i in np.unique(x):
        print(i)
        print(y[np.where(x==i)])


def q1a():
    x1 = np.array([0, 0, 0, 1, 0, 1, 0,])
    x2 = np.array([0, 1, 0, 0, 1, 1, 1,])
    x3 = np.array([1, 0, 1, 0, 1, 0, 0,])
    x4 = np.array([0, 0, 1, 1, 0, 0, 1,])
    y =  np.array([0, 0, 1, 1, 0, 0, 0,])
    
    print('H(y): ', dtree.calc_entropy(dtree.calc_discrete_probability(y)))
    print('x1 gain:', dtree.calc_gain(x1, y))
    print('x2 gain:', dtree.calc_gain(x2, y))
    print('x3 gain:', dtree.calc_gain(x3, y))
    print('x4 gain:', dtree.calc_gain(x4, y))
    
    # H(y):  0.863120568566631
    # x1 gain: 0.061743357932800724
    # x2 gain: 0.46956521111470695
    # x3 gain: 0.0059777114237739015
    # x4 gain: 0.46956521111470695
    # x2 = x4, for maximum gain, so choose x2 as it comes first
    # if x2 = 1, then y = 0 (all x2 when 1 correspond to y when 0)
    # this results in the following data set, where x2 = 0
    
    x1 = np.array([0, 0, 1,])
    x3 = np.array([1, 1, 0,])
    x4 = np.array([0, 1, 1,])
    y =  np.array([0, 1, 1,])
    
    print('H(y): ', dtree.calc_entropy(dtree.calc_discrete_probability(y)))
    print('x1 gain:', dtree.calc_gain(x1, y))
    print('x3 gain:', dtree.calc_gain(x3, y))
    print('x4 gain:', dtree.calc_gain(x4, y))
    
    # H(y):  0.9182958340544896
    # x1 gain: 0.2516291673878229
    # x3 gain: 0.2516291673878229
    # x4 gain: 0.9182958340544896
    # Clearly want to split on x4
    # When x4 = 0 then y = 0, and when x4 = 1 then y = 1
    # Rules summary:
    # If x2 = 1, then y = 0
    # If x2 = 0, then y = 0 if x4 = 0, and y = 1 if x4 = 1


def q2a():
    o = np.array(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r',]) # outlook
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's',]) # wind
    p = np.array(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-',]) # play?
    

    # This is all done with entropy
    print('play? entropy: ', dtree.calc_entropy(dtree.calc_discrete_probability(p)))
    print('outlook gain:', dtree.calc_gain(o, p))
    print('temperature gain:', dtree.calc_gain(t, p))
    print('humidity gain:', dtree.calc_gain(h, p))
    print('windy gain:', dtree.calc_gain(w, p))

    # entropy results
    # play? entropy:  0.9402859586706309
    # outlook gain: 0.2467498197744391
    # temperature gain: 0.029222565658954647
    # humidity gain: 0.15183550136234136
    # windy gain: 0.04812703040826927

    # This is all done with ME
    o = np.array(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r',]) # outlook
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's',]) # wind
    p = np.array(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-',]) # play?

    print('play? ME: ', dtree.calc_majority_error(dtree.calc_discrete_probability(p)))
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_majority_error))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_majority_error))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_majority_error))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_majority_error))

    # play? ME:  0.35714285714285715
    # outlook gain: 0.0714285714285714
    # temperature gain: 5.551115123125783e-17
    # humidity gain: 0.07142857142857145
    # windy gain: 5.551115123125783e-17

    print_vals_of_unique_labels(p, o)

    # Can split on either outlook or humidity, will split on outlook for consistency
    # o
    # [1 1 1 1]
    # r
    # [1 1 0 1 0]
    # s
    # [0 0 0 1 1]
    # Clearly, when outlook = overcast, then play = +

    # Leaf when outlook = r
    o = np.array(['r', 'r', 'r', 'r', 'r',]) # outlook
    t = np.array(['m', 'c', 'c', 'm', 'm',]) # temperature
    h = np.array(['h', 'n', 'n', 'n', 'h',]) # humidity
    w = np.array(['w', 'w', 's', 'w', 's',]) # wind
    p = np.array(['+', '+', '-', '+', '-',]) # play?

    print('play? when o = r ME: ', dtree.calc_majority_error(dtree.calc_discrete_probability(p)))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_majority_error))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_majority_error))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_majority_error))

    # play? when o = r ME:  0.4
    # temperature gain: 0.0
    # humidity gain: 0.0
    # windy gain: 0.4

    # Clearly, split on windy
    # Can see from data, when windy = w, then play? = +, and when windy = s, then play = -

    # Leaf when outlook = s
    o = np.array(['s', 's', 's', 's', 's',]) # outlook
    t = np.array(['h', 'h', 'm', 'c', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'n', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 's',]) # wind
    p = np.array(['-', '-', '-', '+', '+',]) # play?

    # play? when o = s ME:  0.4
    # temperature gain: 0.2
    # humidity gain: 0.4
    # windy gain: 0.0

    # Clearly, split on humidity
    # Can see from data, when humidity = h, then play? = -, and when humidity = n, then play? = +

    print('play? when o = s ME: ', dtree.calc_majority_error(dtree.calc_discrete_probability(p)))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_majority_error))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_majority_error))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_majority_error))



def q2b():
    # This is all done with GI
    o = np.array(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r',]) # outlook
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's',]) # wind
    p = np.array(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-',]) # play?

    print('play? GI: ', dtree.calc_gini_index(dtree.calc_discrete_probability(p)))
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_gini_index))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_gini_index))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_gini_index))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_gini_index))

    # play? GI:  0.4591836734693877
    # outlook gain: 0.11632653061224485
    # temperature gain: 0.018707482993197244
    # humidity gain: 0.09183673469387743
    # windy gain: 0.030612244897959162

    print_vals_of_unique_labels(p, o)

    # outlook has highest gain, so split there
    # o
    # [1 1 1 1]
    # r
    # [1 1 0 1 0]
    # s
    # [0 0 0 1 1]
    # Clearly, when outlook = overcast, then play = +
    
    # Leaf when outlook = r
    o = np.array(['r', 'r', 'r', 'r', 'r',]) # outlook
    t = np.array(['m', 'c', 'c', 'm', 'm',]) # temperature
    h = np.array(['h', 'n', 'n', 'n', 'h',]) # humidity
    w = np.array(['w', 'w', 's', 'w', 's',]) # wind
    p = np.array(['+', '+', '-', '+', '-',]) # play?
    
    print('play? when o = r GI: ', dtree.calc_majority_error(dtree.calc_discrete_probability(p)))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_gini_index))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_gini_index))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_gini_index))

    # play? when o = r GI:  0.4
    # temperature gain: 0.013333333333333308
    # humidity gain: 0.013333333333333308
    # windy gain: 0.48

    # Clearly, split on windy
    # Can see from data, when windy = w, then play? = +, and when windy = s, then play = -
    
    # Leaf when outlook = s
    o = np.array(['s', 's', 's', 's', 's',]) # outlook
    t = np.array(['h', 'h', 'm', 'c', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'n', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 's',]) # wind
    p = np.array(['-', '-', '-', '+', '+',]) # play?
    
    print('play? when o = s GI: ', dtree.calc_majority_error(dtree.calc_discrete_probability(p)))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_gini_index))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_gini_index))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_gini_index))

    # play? when o = s ME:  0.4
    # temperature gain: 0.27999999999999997
    # humidity gain: 0.48
    # windy gain: 0.013333333333333308

    # Clearly, split on humidity
    # Can see from data, when humidity = h, then play? = -, and when humidity = n, then play? = +



def q3a():
    o = np.array(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r', np.nan,]) # outlook
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's', 'w',]) # wind
    p = np.array(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-', '+',]) # play?

    # part a - use most common val in training data as missing val
    o = pd.Series(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r', np.nan,]) # outlook
    missing_val = o.mode()[0]
    o = o.fillna(missing_val)

    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_entropy))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))

    # outlook gain: 0.2273273022811375
    # temperature gain: 0.032498735534292944
    # humidity gain: 0.168621667532054
    # windy gain: 0.0597731301493174


def q3b():
    # part b - use most common val with subset of data with same outcome (i.e. Play = '+')
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's', 'w',]) # wind

    o = pd.Series(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r', np.nan,]) # outlook
    p = pd.Series(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-', '+',]) # play?
    missing_val = o[p=='+'].mode()[0]
    o = o.fillna(missing_val)

    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_entropy))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))

    # outlook gain: 0.27099543775137724
    # temperature gain: 0.032498735534292944
    # humidity gain: 0.168621667532054
    # windy gain: 0.0597731301493174


def q3c():
    # part c - use fractional counts (taken care of in error_calcs)
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's', 'w',]) # wind
    
    o = pd.Series(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r', np.nan,]) # outlook
    p = pd.Series(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-', '+',]) # play?
    
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_entropy))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))
    
    # outlook gain: 0.22444415770980708
    # temperature gain: 0.032498735534292944
    # humidity gain: 0.168621667532054
    # windy gain: 0.0597731301493174


def q3c():
    # part d - use fractional counts and build tree
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h', 'n',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's', 'w',]) # wind
    o = pd.Series(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r', np.nan,]) # outlook
    p = pd.Series(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-', '+',]) # play?
    
    print('\nDetermine first split')
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_entropy))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))
    
    # outlook gain: 0.22444415770980708
    # temperature gain: 0.032498735534292944
    # humidity gain: 0.168621667532054
    # windy gain: 0.0597731301493174

    # obviously, split on outlook
    
    # o
    # ['o', 'o', 'o', 'o', np.nan,]) # outlook
    # ['+', '+', '+', '+', '+',]) # play?

    # s
    # ['s', 's', 's', 's', 's', np.nan,]) # outlook
    # ['-', '-', '-', '+', '+', '+',]) # play?

    # r
    # ['r', 'r', 'r', 'r', 'r', np.nan,]) # outlook
    # ['+', '+', '-', '+', '-', '+',]) # play?

    # Clearly, when outlook = overcast, then play

    # Leaf s of Outlook node
    t = pd.Series(['h', 'h', 'm', 'c', 'm',]) # temperature
    h = pd.Series(['h', 'h', 'h', 'n', 'n',]) # humidity
    w = pd.Series(['w', 's', 'w', 'w', 's',]) # wind
    o = pd.Series(['s', 's', 's', 's', 's',]) # outlook
    p = pd.Series(['-', '-', '-', '+', '+',]) # play?

    print('\nLeaf s of Outlook node')
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))

    # temperature gain: 0.5709505944546686
    # humidity gain: 0.9709505944546686
    # windy gain: 0.01997309402197489

    # Clearly split on humidity
    # When humidity = h, then -, if humidity = n, then +

    # Leaf r of Outlook node
    t = pd.Series(['m', 'c', 'c', 'm', 'm',]) # temperature
    h = pd.Series(['h', 'n', 'n', 'n', 'h',]) # humidity
    w = pd.Series(['w', 'w', 's', 'w', 's',]) # wind
    o = pd.Series(['r', 'r', 'r', 'r', 'r',]) # outlook
    p = pd.Series(['+', '+', '-', '+', '-',]) # play?

    print('\nLeaf r of Outlook node')
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_entropy))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_entropy))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_entropy))

    # temperature gain: 0.01997309402197489
    # humidity gain: 0.01997309402197489
    # windy gain: 0.9709505944546686

    # Clearly split on windy
    # When windy = w, then +, when windy = s then -
