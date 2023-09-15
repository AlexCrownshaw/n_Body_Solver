# n_Body_Solver
The n_Body_Solver uses a direct numerical simulation approach to solve the infamous n body problem. 
A 4th order Runge-Kutta numerical integration method is used for accurate and low computation analysis. 
## Installation
Latest release version 0.1:
```commandline
pip install git+https://github.com/AlexCrownshaw/n_Body_Solver.git@master
```
## Clone
```commandLine
git clone https://github.com/AlexCrownshaw/n_Body_Solver.git
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
ITERATIONS = 40000
DT = 3.154e4 * 2  # two thousandth of a year in seconds

MASS_345 = 1e19
""" PARAMETERS STOP """

def main():
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
    main()
```
![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/3n_40e3iter_2523136920et_15-09-23_17-39-54/Plots/Solution_Animation_3n.gif "Three body Solution")
### Orbit Example

```python
import numpy as np

from n_body_solver.solver import Solver
from n_body_solver.body import Body

""" PARAMETERS START """
ITERATIONS = 32000
DT = 4
""" PARAMETERS STOP """


def main():
    """
    A stationary large mass body is placed at the simulation origin.
    An orbiting body with insignificant mass is placed some distance from the origin with a velocity
    vector tangential to the displacement vector
    """
    
    # Stationary large body
    n1 = Body(m=5.972e24, x=[0, 0, 0])

    # orbiting small body
    orbital_velocity = 7.3501e3
    orbit_radius = 7378.14e3
    n2 = Body(m=1e3, x=[orbit_radius, 0, 0], v=[0, orbital_velocity, 0])

    solver = Solver(bodies=[n1, n2], iterations=30e3, dt=100)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(show=True)
    results.animate_solution()

    
if __name__ == "__main__":
    main()
```

![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/2n_10e3iter_99990et_15-09-23_19-27-33/Plots/Solution_Animation_2n.gif "TOrbit Solution")

### Loading Solutions

```python
from n_body_solver.results import Results

def main():
    # Load and plot solution
    results = Results(solution_path=r"n_body_solver/solutions/3n_40e3iter_2523136920et_15-09-23_17-39-54")
    results.plot_trajectory()
    results.animate_solution()

    # Recover solver from solution and continue simulation
    solver = results.recover_solver()
    solver.config_solver(iterations=1000, dt=DT)
    new_results = solver.solve()

    # Save and plot additional simulated data
    new_results.save_solution()
    new_results.plot_trajectory()
    new_results.plot_velocity()
    new_results.animate_solution(frames=1e6)


if __name__ == "__main__":
    main()
```
