"""
Task

Jordan has $100 to buy some comic books. He really likes the Star Wars books
which cost $12 each, but he could also buy the Marvels books which cost $5 each.
If he has to buy at least 12 books, what is the maximum number of the Star Wars
books that he can buy?
"""

"""
Let s be the number of the Star Wars books and m be the number of the Marvels books

maximize: s
subject to:
    s * 12 + m * 5 <= 100
    s + m >= 12
    s >= 0
    m >= 0
"""

"""
Matrix version

maximize: transpose(c) * x
subject to:
    Ax <= b
    x >= 0

x = [s, m]
c = [1, 0]

# if the first case <= and the second >=
A = [[12, 5], [-1, -1]]
b = [100, -12]
"""

from scipy.optimize import linprog
import pulp


def linear_programming():
    """
    NOTE: this is a linear programming solution, not a linear _integer_
    programming so results are floats!
    """
    print('\n---Solving linear program---')

    # NOTE linprog minimizes a linear objective function!
    # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linprog.html
    c = [-1, 0]
    A = [[12, 5], [-1, -1]]
    b = [100, -12]

    # x >= 0
    s_bounds = (0, None)
    m_bounds = (0, None)

    result = linprog(
        c, A_ub=A, b_ub=b, bounds=(s_bounds, m_bounds), options={"disp": True}
    )

    print('The number of the Star Wars books:', result['x'][0])


def linear_integer_programming():
    print('\n---Solving linear integer program---')

    prob = pulp.LpProblem('books', pulp.LpMaximize)

    # variables
    s = pulp.LpVariable('s', 0, cat='Integer')
    m = pulp.LpVariable('m', 0, cat='Integer')

    # objective
    prob += s

    # constraints

    prob += (s * 12 + m * 5 <= 100)
    prob += (s + m >= 12)

    prob.solve()

    # for v in prob.variables():
        # print('{}={}'.format(v.name, v.varValue))

    print('The number of the Star Wars books:', pulp.value(prob.objective))


if __name__ == '__main__':

    linear_programming()

    linear_integer_programming()
