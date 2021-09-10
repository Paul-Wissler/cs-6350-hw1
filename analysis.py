# This file is only for running calculations for part 1 of the assignment
import DecisionTree as dtree
import numpy as np


def main():
    # x1 = np.array([0, 0, 0, 1, 0, 1, 0,])
    # x2 = np.array([0, 1, 0, 0, 1, 1, 1,])
    # x3 = np.array([1, 0, 1, 0, 1, 0, 0,])
    # x4 = np.array([0, 0, 1, 1, 0, 0, 1,])
    # y =  np.array([0, 0, 1, 1, 0, 0, 0,])

    # print('H(y): ', dtree.calc_entropy(dtree.calc_bool_probability(y)))
    # print('x1 gain:', dtree.calc_gain(x1, y))
    # print('x2 gain:', dtree.calc_gain(x2, y))
    # print('x3 gain:', dtree.calc_gain(x3, y))
    # print('x4 gain:', dtree.calc_gain(x4, y))

    # H(y):  0.863120568566631
    # x1 gain: 0.061743357932800724
    # x2 gain: 0.46956521111470695
    # x3 gain: 0.0059777114237739015
    # x4 gain: 0.46956521111470695

    # x2 = x4, for maximum gain, so choose x2 as it comes first
    # if x2 = 1, then y = 0 (all x2 when 1 correspond to y when 0)
    # this results in the following data set, where x2 = 0

    # x1 = np.array([0, 0, 1,])
    # x3 = np.array([1, 1, 0,])
    # x4 = np.array([0, 1, 1,])
    # y =  np.array([0, 1, 1,])

    # print('H(y): ', dtree.calc_entropy(dtree.calc_bool_probability(y)))
    # print('x1 gain:', dtree.calc_gain(x1, y))
    # print('x3 gain:', dtree.calc_gain(x3, y))
    # print('x4 gain:', dtree.calc_gain(x4, y))

    # H(y):  0.9182958340544896
    # x1 gain: 0.2516291673878229
    # x3 gain: 0.2516291673878229
    # x4 gain: 0.9182958340544896

    # Clearly want to split on x4
    # When x4 = 0 then y = 0, and when x4 = 1 then y = 1

    # Rules summary:
    # If x2 = 1, then y = 0
    # If x2 = 0, then y = 0 if x4 = 0, and y = 0 if x4 = 1

    # Q2
    o = np.array(['s', 's', 'o', 'r', 'r', 'r', 'o', 's', 's', 'r', 's', 'o', 'o', 'r',]) # outlook
    t = np.array(['h', 'h', 'h', 'm', 'c', 'c', 'c', 'm', 'c', 'm', 'm', 'm', 'h', 'm',]) # temperature
    h = np.array(['h', 'h', 'h', 'h', 'n', 'n', 'n', 'h', 'n', 'n', 'n', 'h', 'n', 'h',]) # humidity
    w = np.array(['w', 's', 'w', 'w', 'w', 's', 's', 'w', 'w', 'w', 's', 's', 'w', 's',]) # wind
    p = np.array(['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-',]) # play?
    print(p)
    p[np.where(p=='-')] = 0
    p[np.where(p=='+')] = 1
    p = p.astype(int)
    print(p)

    # This is all done with entropy
    print('play? entropy: ', dtree.calc_entropy(dtree.calc_bool_probability(p)))
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
    print('play? ME: ', dtree.calc_majority_error(dtree.calc_bool_probability(p)))
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_majority_error))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_majority_error))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_majority_error))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_majority_error))

    # play? ME:  0.35714285714285715
    # outlook gain: 0.0714285714285714
    # temperature gain: 5.551115123125783e-17
    # humidity gain: 0.07142857142857145
    # windy gain: 5.551115123125783e-17

    # This is all done with GI
    print('play? GI: ', dtree.calc_gini_index(dtree.calc_bool_probability(p)))
    print('outlook gain:', dtree.calc_gain(o, p, f=dtree.calc_gini_index))
    print('temperature gain:', dtree.calc_gain(t, p, f=dtree.calc_gini_index))
    print('humidity gain:', dtree.calc_gain(h, p, f=dtree.calc_gini_index))
    print('windy gain:', dtree.calc_gain(w, p, f=dtree.calc_gini_index))

    # play? GI:  0.4591836734693877
    # outlook gain: 0.11632653061224485
    # temperature gain: 0.018707482993197244
    # humidity gain: 0.09183673469387743
    # windy gain: 0.030612244897959162

main()
