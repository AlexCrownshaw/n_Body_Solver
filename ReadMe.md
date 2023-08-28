##Usage examples

```python
from n_body_solver.solver import Solver
from n_body_solver.body import Body

""" PARAMETERS START """
ITERATIONS = 3000
DT = 0.1
""" PARAMETERS STOP """


def three_body():
    n1 = Body(m=10, x=[-10e6, 0, 0], v=[0, -10, 0])
    n2 = Body(m=1e24, x=[0, 0, 0], v=[0, 0, 0])
    n3 = Body(m=10, x=[10e6, 0, 0], v=[0, 10, 0])

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.plot_trajectory(show=True)
    print(solver, results)


def orbit():
    n1 = Body(m=5.972e24, x=[0, 0, 0])
    n2 = Body(m=1e3, x=[600e3, 0, 0], v=[0, 7.56e3, 0])

    solver = Solver(bodies=[n1, n2], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.plot_trajectory(show=True)


if __name__ == "__main__":
    three_body()
    orbit()

```
