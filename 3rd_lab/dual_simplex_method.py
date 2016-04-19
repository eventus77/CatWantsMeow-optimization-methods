# coding=utf-8
import numpy as np


class DualSimplexMethod(object):
    max_iterations = 300
    unbounded_function_error = "Ограничения исходной задачи несовместны"
    singular_matrix_error = "Матрица является вырожденной"

    def second_phase(self, a, b, c, y, j_basis):
        n, m = len(c), len(b)

        try:
            # counting inverted matrix first time
            a_basis_inv = np.linalg.inv(a.take(j_basis, 1))
        except np.linalg.LinAlgError:
            # raise error if matrix is singular
            raise RuntimeError(self.singular_matrix_error)

        for i in xrange(self.max_iterations):
            # counting coprogram and non basis indexes
            j_non_basis = list(set(range(n)).difference(j_basis))
            delta = np.dot(y, a) - c

            # counting pseudoprogram and checking if it is optimal
            px = np.dot(a_basis_inv, b)
            if np.all(px >= 0):
                x = np.array([0] * len(c), dtype=float)
                for i, value in zip(j_basis, px):
                    x[i] = value
                return x, j_basis

            # counting mu. checking if function is unbounded
            s = np.where(px < 0)[0][0]
            mu = np.dot(a_basis_inv[s], a)
            if np.all(mu >= 0):
                raise RuntimeError(self.unbounded_function_error)

            # counting sigms
            jo = min(np.where(mu < 0)[0], key=lambda j: -delta[j] / mu[j])
            sigma = -delta[jo] / mu[jo]

            # updating basis and basis indexes
            y = y + sigma * a_basis_inv[s]
            j_basis[s] = jo

            # counting z vector. updating inverted basis matrix
            z = np.dot(a_basis_inv, a[:, jo])
            t = np.eye(m, m)
            t[:, s] = -np.array(list(z[: s]) + [-1] + list(z[s + 1:])) / z[s]
            a_basis_inv = np.dot(t, a_basis_inv)


def main():
    a = np.array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [4, 2, 1, 0, 1, 5, -1, -5],
        [1, 1, 0, -1, 0, 3, -1, 1],
    ])

    c = np.array([5, 2, 3, -16, 1, 3, -3, -12], dtype=float)
    b = np.array([-2, -4, -2], dtype=float)
    y = np.array([1, 2, -1], dtype=float)
    j = [0, 1, 2]

    try:
        method = DualSimplexMethod()
        x, j_basis = method.second_phase(a, b, c, y, j)
        print "Базисный план: ({})".format(", ".join(map(str, x)))
        print "Базисные индексы: {0}".format(", ".join(map(str, j_basis)))
        print "Оптимальное значение целевой функции:", np.dot(c, x)
    except RuntimeError as e:
        print e.message


if __name__ == '__main__':
    main()
