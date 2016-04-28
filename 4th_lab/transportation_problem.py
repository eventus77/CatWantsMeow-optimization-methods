from copy import copy, deepcopy


class Math(object):

    @staticmethod
    def zeros(n, m):
        """ Generate nxm dimensions array filled by zeros """
        array = []
        for i in xrange(n):
            array.append([0 for j in xrange(m)])
        return array

    @staticmethod
    def pmin(*args):
        """ Get minimal positive value from passed arguments """
        return max(min(*args), 0)

    @staticmethod
    def prettify(array):
        """ Get pretty string interpretation of two-dimensional array """
        return '\n'.join(''.join("{:>4}".format(elem) for elem in row) for row in array)


class TranshipmentProblem(object):

    eps = 0.00001

    def __init__(self, supply, demand, cost):
        self.supply = deepcopy(supply)
        self.demand = deepcopy(demand)
        self.cost = deepcopy(cost)

    def _balance(self):
        """ Added additional value of supply or demand if sum of supply and demand is not equal """
        total_supply = sum(self.supply)
        total_demand = sum(self.demand)
        if total_supply > total_demand:
            self.demand.append(total_supply - total_demand)
        elif total_demand > total_supply:
            self.supply.append(total_demand - total_supply)

    def _prevent_degeneracy(self):
        """Add a small amount to demand and total supply to prevent degeneracy"""
        self.demand = [d + self.eps for d in self.demand]
        self.supply[0] += len(self.demand) * self.eps

    def _get_initial_solution(self):
        """ Calculation of initial solution by north-west algorithm """
        supply, demand = copy(self.supply), copy(self.demand)
        amounts = Math.zeros(len(supply), len(demand))
        for i, provider in enumerate(supply):
            for j, consumer in enumerate(demand):
                if supply[i] <= 0:
                    break
                amounts[i][j] = Math.pmin(supply[i], demand[j])
                supply[i] -= amounts[i][j]
                demand[j] -= amounts[i][j]
        return amounts


    def solve():
        pass


def main():
    supply = [20, 30, 25]
    demand = [10, 10, 10, 10, 10]
    cost = [
        [2, 8, -5, 7, 10],
        [11, 5, 8, -8, 4],
        [1, 3, 7, 4, 2],
    ]

    TranshipmentProblem._balance(supply, demand)
    amounts = TranshipmentProblem._get_initial_solution(supply, demand)
    print TranshipmentProblem._check_consistence(amounts)


if __name__ == '__main__':
    main()
