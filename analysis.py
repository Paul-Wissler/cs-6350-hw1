# This file is only for running calculations for part 1 of the assignment
import DecisionTree as dtree
import numpy as np


def main():
    x1 = np.array([0, 0, 0, 1, 0, 1, 0,])
    x2 = np.array([0, 1, 0, 0, 1, 1, 1,])
    x3 = np.array([1, 0, 1, 0, 1, 0, 0,])
    x4 = np.array([0, 0, 1, 1, 0, 0, 1,])
    y = np.array([0, 0, 1, 1, 0, 0, 0,])

    print('H(y): ', dtree.calc_entropy(dtree.calc_bool_probability(y)))
    print('x1 gain:', dtree.calc_gain(x1, y))
    print('x2 gain:', dtree.calc_gain(x2, y))
    print('x3 gain:', dtree.calc_gain(x3, y))
    print('x4 gain:', dtree.calc_gain(x4, y))

    # H(y):  0.863120568566631
    # x1 gain: 0.061743357932800724
    # x2 gain: 0.46956521111470695
    # x3 gain: 0.0059777114237739015
    # x4 gain: 0.46956521111470695

main()