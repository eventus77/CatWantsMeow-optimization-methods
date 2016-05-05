# coding=utf-8
from numpy import array, zeros
from numpy.linalg import solve


def south_west_angle_method(supply, demand):
    i, j = 0, 0
    supply, demand = array(supply), array(demand)
    basis_coords = list()
    amounts = zeros([len(supply), len(demand)])
    while i != len(supply) and j != len(demand):
        basis_coords.append((i, j))
        if demand[j] < supply[i]:
            amounts[i, j] = demand[j]
            supply[i] -= demand[j]
            j += 1
        elif supply[i] < demand[j]:
            amounts[i, j] = supply[i]
            demand[j] -= supply[i]
            i += 1
        else:
            amounts[i, j] = supply[i]
            i += 1
            if i == len(supply):
                break
            basis_coords.append((i, j))
            j += 1
    return amounts, array(basis_coords)


def get_neighbours(amounts, i, j, basis_coords, is_new=False):
    neighbours = list()

    l = j
    k = i - 1
    while k >= 0:
        if (k, l) in basis_coords:
            neighbours.append((k, l))
            break
        k -= 1

    k = i
    l = j - 1
    while l >= 0:
        if (k, l) in basis_coords:
            neighbours.append((k, l))
            break
        l -= 1
    k = i + 1
    l = j

    if is_new and len(neighbours) == 2:
        return neighbours

    while k < amounts.shape[0]:
        if (k, l) in basis_coords:
            neighbours.append((k, l))
            break
        k += 1
    k = i
    l = j + 1

    if is_new and len(neighbours) == 2:
        return neighbours
    while l < amounts.shape[1]:
        if (k, l) in basis_coords:
            neighbours.append((k, l))
            break
        l += 1
    return neighbours


def get_corner_peaks(path):
    corner_peaks = list()
    length = len(path)
    for k, peak in enumerate(path):
        i = (k + 1) % length
        j = (k - 1 + length) % length
        if path[i][0] != path[j][0] and path[i][1] != path[j][1]:
            corner_peaks.append(peak)
    return corner_peaks


def dfs(v, ancestors, adjacency_list):
    for neighbour in adjacency_list[v]:
        if neighbour == ancestors[v]:
            continue
        else:
            ancestors[neighbour] = v
            dfs(neighbour, ancestors, adjacency_list)


def get_cycle(amounts, basis_coords, i0, j0):
    coords_tuple = tuple(tuple(coord) for coord in basis_coords)
    ancestors = {k: [] for k in coords_tuple}

    adjacency_list = dict()
    adjacency_list[i0, j0] = get_neighbours(amounts, i0, j0, coords_tuple, True)
    for i in xrange(amounts.shape[0]):
        for j in xrange(amounts.shape[1]):
            if (i, j) in coords_tuple:
                adjacency_list[(i, j)] = get_neighbours(amounts, i, j, coords_tuple)

    dfs(adjacency_list[(i0, j0)][0], ancestors, adjacency_list)

    path = list()
    start = adjacency_list[(i0, j0)][0]
    current = adjacency_list[(i0, j0)][1]
    while current != start:
        path.append(current)
        current = ancestors[current]
    path.append(start)
    path.insert(0, (i0, j0))
    return get_corner_peaks(path)


def matrix_transport_problem(amounts, basis_coords, costs):
    while True:
        equations = zeros([basis_coords.shape[0] + 1, costs.shape[0] + costs.shape[1]])
        b = zeros(basis_coords.shape[0] + 1)
        for k, (i, j) in enumerate(basis_coords):
            equations[k, i], equations[k, j + costs.shape[0]] = 1, 1
            b[k] = costs[i, j]
        equations[basis_coords.shape[0], 0] = 1
        b[basis_coords.shape[0]] = 0

        results = solve(equations, b)
        u, v = results[:costs.shape[0]], results[costs.shape[0]:]

        coords_tuple = tuple(tuple(coord) for coord in basis_coords)
        non_basis_coords = [
            (i, j)
            for i in xrange(costs.shape[0])
            for j in xrange(costs.shape[1])
            if (i, j) not in coords_tuple
        ]

        if all(costs[i, j] > u[i] + v[j] for i, j in non_basis_coords):
            return amounts, basis_coords

        old_coord = min(filter(lambda (i, j): u[i] + v[j] > costs[i, j], non_basis_coords))
        cycle = get_cycle(amounts, basis_coords, *old_coord)
        marks = {'+': cycle[::2], '-': cycle[1::2]}

        min_coords = array([amounts[i, j] for i, j in marks['-']])
        theta = min_coords.min()
        new_coord = marks['-'][min_coords.argmin()]

        for i, j in cycle:
            amounts[i, j] += theta if (i, j) in marks['+'] else -theta

        basis_coords = list([tuple(coord) for coord in basis_coords])
        basis_coords.append(old_coord)
        basis_coords.remove(new_coord)
        basis_coords = array(basis_coords)


def main():
    supply = array([100, 50, 80])
    demand = array([70, 70, 90])
    cost = array([
        [2, 3, 4],
        [6, 1, 5],
        [6, 4, 2]
    ])

    initial_amounts, basis_coords = south_west_angle_method(supply, demand)
    optimal_amounts, basis_coords = matrix_transport_problem(initial_amounts, basis_coords, cost)

    transhipment_costs = 0
    for i, row in enumerate(optimal_amounts):
        for j, elem in enumerate(row):
            transhipment_costs += cost[i, j] * optimal_amounts[i, j]

    pretty_amounts = "\n".join("".join("{:<8}".format(elem) for elem in row) for row in optimal_amounts)
    print "Оптимальный план перевозок:\n", pretty_amounts
    print "Затраты на перевозку:", transhipment_costs


if __name__ == "__main__":
    main()
