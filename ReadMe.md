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
from n_body_solver import Solver
from n_body_solver import Body
from n_body_solver import Constants

""" PARAMETERS START """
ITERATIONS = 40000
DT = Constants.TIME_UNITS["year"] * 4  # four thousandth of a year in seconds
""" PARAMETERS STOP """


def main():
    n1 = Body(m=59.5, x=[-5, 9, 0], v=[4.911, -3.13, 0], m_unit="sm", x_unit="au", v_unit="kmps")
    n2 = Body(m=142.2, x=[-33, -20, 0], v=[1.839, -5.072, 0], m_unit="sm", x_unit="au", v_unit="kmps")
    n3 = Body(m=41.7, x=[4, -32, 0], v=[-4.533, 3.062, 0], m_unit="sm", x_unit="au", v_unit="kmps")

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(save=True)
    results.animate_solution(frames=100)
    results.plot_velocity()


if __name__ == "__main__":
    main()
```
![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/3n_40e3iter_5046273840et_24-09-23_14-24-04/Plots/Solution_Animation_3n.gif "Three body Solution")
### Orbit Example

```python
import numpy as np

from n_body_solver import Solver
from n_body_solver import Body
from n_body_solver import Constants


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
    orbit_radius = 7e6
    orbital_velocity = np.sqrt((Constants.G * n1.m) / orbit_radius)
    n2 = Body(m=1, x=[orbit_radius, 0, 0], v=[0, orbital_velocity, 0])

    solver = Solver(bodies=[n1, n2], iterations=20e3, dt=10)
    results = solver.solve()
    results.save_solution()
    results.plot_trajectory(show=True)
    results.animate_solution()


    
if __name__ == "__main__":
    main()
```

![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/2n_20e3iter_199990et_24-09-23_14-42-17/Plots/Solution_Animation_2n.gif "TOrbit Solution")

### Loading Solutions

```python
from n_body_solver import Results

def main():
    # Load and plot solution
    results = Results(solution_path=r"n_body_solver/solutions/3n_40e3iter_2523136920et_15-09-23_17-39-54")
    results.plot_trajectory()
    results.animate_solution()

    # Recover solver from solution and continue simulation
    solver = results.recover_solver()
    solver.config_solver(iterations=1000, dt=1)
    new_results = solver.solve()

    # Save and plot additional simulated data
    new_results.save_solution()
    new_results.plot_trajectory()
    new_results.plot_velocity()
    new_results.animate_solution(frames=1e6)


if __name__ == "__main__":
    main()
```
