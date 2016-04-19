# coding=utf-8
import numpy as np


class SimplexMethod(object):

    max_iterations = 3000
    unbounded_function_error = "Целевая функция не ограничена сверху на множестве планов"
    singular_matrix_error = "Матрица является вырожденной"

    def second_phase(self, a, b, c, x, j_basis):
        n, m = len(c), len(b)

        try:
            # counting inverted matrix first time
            a_basis_inv = np.linalg.inv(a.take(j_basis, 1))
        except np.linalg.LinAlgError:
            # raise error if matrix is singular
            raise RuntimeError(self.singular_matrix_error)

        for i in xrange(self.max_iterations):
            j_non_basis = list(set(range(n)).difference(j_basis))

            # counting potentials and estimates vector
            u = np.dot(c.take(j_basis), a_basis_inv)
            deltas = np.dot(u, a) - c
            if np.all(deltas >= 0):
                return x, j_basis

            # counting z. checking if function is unbounded
            jo = min(np.where(deltas < 0)[0])
            z = np.dot(a_basis_inv, a[:, jo])
            if np.all(z <= 0):
                raise RuntimeError(self.unbounded_function_error)

            # counting theta
            s = min(np.where(z > 0)[0], key=lambda i: x[j_basis[i]] / z[i])
            theta = x[j_basis[s]] / z[s]

            # updating basis program and basis indexes
            x[j_non_basis] = 0
            x[jo] = theta
            x[j_basis] = x[j_basis] - np.dot(z, theta)
            j_basis[s] = jo
            # j_basis.sort()

            # updating inverted basis matrix
            t = np.eye(m, m)
            t[:, s] = -np.array(list(z[: s]) + [-1] + list(z[s+1:])) / z[s]
            a_basis_inv = np.dot(t, a_basis_inv)


def main():
    a = np.array([
        [0, 1, 4, 1, 0, -3, 5, 0],
        [1, -1, 0, 1, 0, 0, 1, 0],
        [0, 7, -1, 0, -1, 3, 8, 0],
        [1, 1, 1, 1, 0, 3, -3, 1],
    ])

    c = np.array([-5, -2, 3, -4, -6, 0, -1, -5], dtype=float)
    b = np.array([6, 10, -2, 15], dtype=float)
    x = np.array([4, 0, 0, 6, 2, 0, 0, 5], dtype=float)
    j = [0, 3, 4, 7]

    try:
        method = SimplexMethod()
        x, j_basis = method.second_phase(a, b, c, x, j)
        print "Базисный план: ({})".format(", ".join(map(str, x)))
        print "Базисные индексы: {0}".format(", ".join(map(str, j_basis)))
    except RuntimeError as e:
        print e.message

if __name__ == '__main__':
    main()
