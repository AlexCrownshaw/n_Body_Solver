# n_Body_Solver
This projects uses a direct numerical simulation approach to the infamous n body problem. 
A 4th order Runge-Kutta numerical integration method is used for accurate and low computation analysis. 
## Installation
Latest release version 0.1:
```commandline
pip install git+https://github.com/AlexCrownshaw/n_Body_Solver.git@master
```
### Install requirements (Only required when cloning)
```commandline
pip install -r requirements.txt
```
## Usage examples

### Three Body Example
Using Astronomical Units and Solar Masses
```python
from n_body_solver.solver import Solver
from n_body_solver.body import Body

""" PARAMETERS START """
ITERATIONS = 10000
DT = 5

MASS_345 = 1e20
""" PARAMETERS STOP """

def three_body():
    n1 = Body(m=138.3, x=[-16, 21, 0], m_unit="sm", x_unit="au")
    n2 = Body(m=46.1, x=[22, -17, 0], m_unit="sm", x_unit="au")
    n3 = Body(m=29.2, x=[-2, -10, 0], m_unit="sm", x_unit="au")

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(save=True)
    results.animate_solution(frames=100)
    results.plot_velocity()


if __name__ == "__main__":
    three_body()()

```
![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/3n_30e3iter_946168460et_10-09-23_22-34-45/Plots/Solution_Animation_3n.gif "Logo Title Text 1")
### Orbit

```python
import numpy as np

from n_body_solver.solver import Solver
from n_body_solver.body import Body

""" PARAMETERS START """
ITERATIONS = 32000
DT = 4
""" PARAMETERS STOP """


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

    
if __name__ == "__main__":
    orbit()
```