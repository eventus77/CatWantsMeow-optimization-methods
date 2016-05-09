# -*- coding: utf-8 -*-
import numpy as np
from numpy import dot
from scipy.optimize import linprog


eps = 10e-3


def main():

    b = [
        np.array([
            [ 2.00,  1.00,  0.00,  4.00,  0.00,  3.00,  0.00,  0.00],
            [ 0.00,  4.00,  0.00,  3.00,  1.00,  1.00,  3.00,  2.00],
            [ 1.00,  3.00,  0.00,  5.00,  0.00,  4.00,  0.00,  4.00],
        ]),
        np.array([
            [ 0.00,  0.00,  0.50,  2.50,  1.00,  0.00, -2.50, -2.00],
            [ 0.50,  0.50, -0.50,  0.00,  0.50, -0.50, -0.50, -0.50],
            [ 0.50,  0.50,  0.50,  0.00,  0.50,  1.00,  2.50,  4.00],
        ]),
        np.array([
            [ 1.00,  2.00, -1.50,  3.00, -2.50,  0.00, -1.00, -0.50],
            [-1.50, -0.50, -1.00, -2.50,  3.50, -3.00, -1.50, -0.50],
            [ 1.50,  2.50, -1.00,  1.00,  2.50,  1.50,  3.00,  0.00],
        ]),
        np.array([
            [ 0.75,  0.50, -1.00,  0.25,  0.25,  0.00,  0.25,  0.75],
            [-1.00,  1.00,  4.00,  0.75,  0.75,  0.50,  7.00, -0.75],
            [ 0.50, -0.25,  0.50,  0.75,  0.50,  1.25, -0.75, -0.25],
        ]),
        np.array([
            [ 1.50, -1.50, -1.50,  2.00,  1.50,  0.00,  0.50, -1.50],
            [-0.50, -2.50, -0.50, -5.00, -2.50,  3.50,  1.00,  2.00],
            [-2.50,  1.00, -2.00, -1.50, -2.50,  0.50,  8.50, -2.50],
        ]),
        np.array([
            [ 1.00,  0.25, -0.50,  1.25,  1.25, -0.50,  0.25, -0.75],
            [-1.00, -0.75, -0.75,  0.50, -0.25,  1.25,  0.25, -0.50],
            [ 0.00,  0.75,  0.50, -0.50, -1.00,  1.00, -1.00,  1.00],
        ])
    ]

    c = [
        np.array([-1.00, -1.00, -1.00, -1.00, -2.00,  0.00, -2.00, -3.00]),
        np.array([ 0.00,  60.0,  80.0,  0.00,  0.00,  0.00,  40.0,  0.00]),
        np.array([ 2.00,  0.00,  3.00,  0.00,  2.00,  0.00,  3.00,  0.00]),
        np.array([ 0.00,  0.00,  80.0,  0.00,  0.00,  0.00,  0.00,  0.00]),
        np.array([ 0.00, -2.00,  1.00,  2.00,  0.00,  0.00, -2.00,  1.00]),
        np.array([-4.00, -2.00,  6.00,  0.00,  4.00, -2.00,  60.0,  2.00]),
    ]

    a = np.array([0.00, -687.125, -666.625, -349.5938, -254.625, -45.15])
    initial_x = np.array([0.00, 8.00, 2.00, 1.00, 0.00, 4.00, 0.00, 0.00])

    f = lambda x: 0.5 * dot(x, dot(b[0].T, dot(b[0], x))) + dot(c[0], x)
    new_x = solve_problem(a, b, c, initial_x)
    print "Решение улучшено: {} -> {}, x = {}".format(f(initial_x), f(new_x), new_x.tolist())


def solve_problem(a, b, c, initial_x):
    f = lambda x: 0.5 * dot(x, dot(b[0].T, dot(b[0], x))) + dot(c[0], x)
    g = lambda x, i: 0.5 * dot(x, dot(b[i].T, dot(b[i], x))) + dot(c[i], x) + a[i]

    derivative = lambda i: dot(initial_x, dot(b[i].T, b[i])) + c[i]
    df = derivative(0)
    dg = [derivative(i) for i in xrange(1, len(c))]

    indexes = [i for i in xrange(len(c) - 1) if abs(g(initial_x, i)) < eps]
    constraints = np.array([dg[i] for i in indexes])
    values = np.zeros(len(indexes))

    low_bound = [0 if abs(x) < eps else -1 for x in initial_x]
    up_bound = [1, 1, 1, 1, 1, 1, 1, 1]
    solution = linprog(df, constraints, values, bounds=zip(low_bound, up_bound))

    l = solution.x
    func_value = dot(df, l)
    if func_value > 0:
        return initial_x

    dx = np.array([0, 0, 0, 0, 0, 0, 0, 0]) - initial_x
    alpha = -0.5 * dot(df, l) / dot(df, dx)

    t = 1
    x = lambda t: initial_x + t * (l + alpha * dx)
    while f(x(t)) >= f(initial_x) and any(g(x(t), i) > 0 for i in xrange(1, len(c))):
        t /= 2
    return x(t)



if __name__ == '__main__':
    main()