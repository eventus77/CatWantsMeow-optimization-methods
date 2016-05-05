# coding=utf-8
import numpy as np

from numpy.linalg import inv
from numpy import dot, array, zeros, ix_, inf, abs


epsilon = 10 ** (-4)


def has_j(basis_indexes, extended_basis_indexes, a_basis_inv, a, j):
    diff = list(set(extended_basis_indexes) - set(basis_indexes))
    for ind in diff:
        if dot(a_basis_inv, a[:, ind])[basis_indexes.index(j)] != 0:
            return True
    return False


def count_direction(basis, a, extend_basis_indexes, d, j0):
    n = len(extend_basis_indexes)
    h = zeros([n + a.shape[0], n + a.shape[0]])
    h[:n, :n] = d[ix_(extend_basis_indexes, extend_basis_indexes)]
    h[n:, :n] = a[:, extend_basis_indexes]
    h[:n, n:] = a[:, extend_basis_indexes].T

    l = zeros(basis.shape[0])
    b = zeros(n + a.shape[0])
    b[:n] = d[ix_(extend_basis_indexes, [j0, ])].T[0]
    b[n:] = a[:, j0]

    l[extend_basis_indexes] = dot(inv(h), -b)[:n]
    l[j0] = 1
    return l


def square_simplex_method(basis, a, basis_indexes, extend_basis_indexes, d, c):
    while True:
        a_basis_inv = inv(a[:, basis_indexes])
        cx = c + dot(d, basis)

        u = -dot(cx[basis_indexes], a_basis_inv)
        delta = dot(u, a) + cx
        j0 = delta.argmin()
        if (delta >= -epsilon).all():
            return basis

        while True:
            l = count_direction(basis, a, extend_basis_indexes, d, j0)
            sigma = dot(l.T, dot(d, l))

            theta = zeros(len(basis))
            theta[:] = inf
            theta[j0] = abs(delta[j0]) / sigma

            for ind in extend_basis_indexes:
                if l[ind] >= -epsilon:
                    theta[ind] = inf
                else:
                    theta[ind] = -basis[ind] / l[ind]

            theta0 = theta.min()
            if theta0 == inf:
                raise Exception('Inconsistent task')
            basis = basis + dot(theta0, l)
            j = theta.argmin()

            if j == j0:
                extend_basis_indexes.append(j)
                break

            elif j in extend_basis_indexes and j not in basis_indexes:
                extend_basis_indexes.remove(j)
                break

            elif (j in basis_indexes and
                          len(extend_basis_indexes) > len(basis_indexes) and
                      has_j(basis_indexes, extend_basis_indexes, a_basis_inv, a, j)):
                diff = list(set(extend_basis_indexes) - set(basis_indexes))
                j_plus = 0
                for ind in diff:
                    if dot(a_basis_inv, a[:, ind])[basis_indexes.index(j)] != 0:
                        j_plus = ind
                extend_basis_indexes.remove(j)
                basis_indexes[basis_indexes.index(j)] = j_plus
                delta[j0] = delta[j0] + theta0 * sigma

            elif (j in extend_basis_indexes and
                      (set(basis_indexes) == set(extend_basis_indexes) or
                                   len(extend_basis_indexes) > len(basis_indexes) and
                                      has_j(basis_indexes, extend_basis_indexes, a_basis_inv, a, j))):
                basis_indexes[basis_indexes.index(j)] = j0
                extend_basis_indexes[extend_basis_indexes.index(j)] = j0
                delta[j0] = delta[j0] + theta0 * sigma


def main():
    a = array([
        [0, 2, 1, 4, 3, 0, -5, -10],
        [-1, 1, 1, 0, 1, 1, -1, -1],
        [1, 1, 1, 0, 1, -2, -5, 8],
    ])
    b = [
        [25, 10, 0, 3, -1, 13, 0, 1],
        [10, 45, 0, 0, 0, 20, 0, 0],
        [0, 0, 20, 0, 0, 0, 0, 0],
        [3, 0, 0, 29, -3, 15, 0, 3],
        [-1, 0, 0, -3, 21, -5, 0, -1],
        [13, 20, 0, 15, -5, 61, 0, 5],
        [0, 0, 0, 0, 0, 0, 20, 0],
        [1, 0, 0, 3, -1, 5, 0, 21],
    ]
    d = array(b) #np.dot(np.transpose(b), b)
    c = array([1, 3, -1, 3, 5, 2, -2, 0])
    #
    a = array([
        [6., 6, 0],
        [3, 0, 1],
    ])
    d = array([
        [4., -2, 0],
        [-2, 4, 0],
        [0, 0, 1],
    ])
    c = array([-1., 0, 0])

    basis_indexes = [1, 2]
    extend_basis_indexes = [1, 2]
    initial_basis = array([0, 1./2, 1])

    optimal_basis = square_simplex_method(initial_basis, a, basis_indexes, extend_basis_indexes, d, c)
    optimal_value = dot(c, optimal_basis) + 0.5 * dot(optimal_basis, dot(d, optimal_basis))
    print "Оптимальный план:", optimal_basis.tolist()
    print "Оптимальное значение целевой функции:", optimal_value


if __name__ == "__main__":
    main()
