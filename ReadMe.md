##Usage examples

```python
iimport numpy as np

from n_body_solver.solver import Solver
from n_body_solver.body import Body

""" PARAMETERS START """
ITERATIONS = 10000
DT = 5

MASS_345 = 1e20
""" PARAMETERS STOP """


def three_body():
    n1 = Body(m=10, x=[-10e6, 0, 0], v=[0, -10, 0])
    n2 = Body(m=1e24, x=[0, 0, 0], v=[0, 0, 0])
    n3 = Body(m=10, x=[10e6, 0, 0], v=[0, 10, 0])

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(show=True)
    print(solver, results)


def orbit():
    # Stationary large body
    n1 = Body(m=5.972e24, x=[0, 0, 0])

    # orbiting small body
    orbital_velocity = 7.56e3
    v_x = orbital_velocity * np.sin(np.deg2rad(45))
    v_y = orbital_velocity * np.cos(np.deg2rad(45))
    n2 = Body(m=1e3, x=[600e3, 0, 0], v=[-v_x, v_y, 0])

    solver = Solver(bodies=[n1, n2], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(show=True)
    results.animate_solution()


def three_four_five():
    n1 = Body(m=MASS_345, x=[300e3, 0, 0])
    n2 = Body(m=MASS_345, x=[0, 400e3, 0])
    n3 = Body(m=MASS_345, x=[0, 0, 0])

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.plot_trajectory()
    results.animate_solution(frames=100)


if __name__ == "__main__":
    three_body()
    orbit()
    three_four_five()

```
