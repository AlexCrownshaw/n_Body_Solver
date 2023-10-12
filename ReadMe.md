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
DT = Constants.TIME_UNITS["year"] / 1e3  # thousandth of a year in seconds

RESULTS_PATH = r"C:\Dev\n_Body_Solver\n_body_solver\solutions"
""" PARAMETERS STOP """


def main():
    n1 = Body(m=59.5, x=[-5, 9, 0], v=[4.911, -3.13, 0], m_unit="sm", x_unit="au", v_unit="kmps")
    n2 = Body(m=142.2, x=[-33, -20, 0], v=[1.839, -5.072, 0], m_unit="sm", x_unit="au", v_unit="kmps")
    n3 = Body(m=41.7, x=[4, -32, 0], v=[-4.533, 3.062, 0], m_unit="sm", x_unit="au", v_unit="kmps")

    solver = Solver(bodies=[n1, n2, n3], iterations=ITERATIONS, dt=DT)
    results = solver.solve()
    results.save_solution(path=RESULTS_PATH)
    results.plot_trajectory(save=True)
    results.animate_solution(frames=100)
    results.plot_velocity()


if __name__ == "__main__":
    main()
```

![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/3n_40e3iter_1261568460et_30-09-23_14-55-05/Plots/Solution_Animation_3n.gif "Three body Solution")
![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/3n_40e3iter_1261568460et_30-09-23_14-55-05/Plots/Velocity_Mag_3n_[0,40000]iter_rng.png "Three body Velocity")

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

### Complex Solver
For the use of complex modules that require some external computation at run time such as a satellite attitude control system, the solver must be configured manually.
```python
import numpy as np

from n_body_solver import Solver
from n_body_solver import Body
from n_body_solver import RBody
from n_body_solver import Constants


""" PARAMETERS START """
ITERATIONS = 32000
DT = 4
""" PARAMETERS STOP """

def main():
    n1 = Body(m=5.972e24, x=[0, 0, 0])

    # orbiting small body
    orbit_radius = 7e6
    orbital_velocity = np.sqrt((Constants.G * n1.m) / orbit_radius)
    satellite = RBody(m=1, x=[orbit_radius, 0, 0], v=[0, orbital_velocity, 0], i=[0.1, 0.1, 0.1])

    solver = Solver(bodies=[n1, satellite], iterations=ITERATIONS, dt=DT)
    solver.init_solver()
    
    # Here a satellite torque vector in cartesian form could be computed by a simulated control system
    T = [10, 0, 0]

    for iteration in range(1, solver.iterations):
        time = iteration * satellite.dt
        satellite.compute_rotation(T=T)
        solver.compute_iteration(i=iteration, t=time)

        solver.print_debug(i=iteration)
        
        
if __name__ == "__main__":
    main()

```

## Using the Quaternion submodule
This project contains an abstract class that can be used to manipulate rotations using the quaternion rotation system
```python
import os

from n_body_solver import Quaternion
from n_body_solver import Results

def main():
    # Define right hand Tait-Bryan Euler angle matrix in degrees
    e = [45, 45, 45]

    # Get quaternion representation of euler angle
    q = Quaternion.from_euler(e=e)
    
    # Plot rotation as a cartesian coordinate system
    Quaternion.plot_quaternion(q=q)
    
    # Animate rotation using quaternion data. To do this we will load a solution. The solution in questions shows an
    # under-damped PID loop commanding a psi (about Z) rotation by 90 degrees"""
    solution_path = r"n_body_solver/solutions/2n_1e3iter_100et_12-10-23_12-58-38"
    results = Results(solution_path=solution_path)

    # The first 300 iterations of the quaternion rotation data are isolated, then animated and saved.
    q_data = results.bodies[1].get_quaternion_data(iter_range=[0, 300])
    Quaternion.animate_rotation(q_data=q_data, frames=300, path=os.path.join(solution_path, "Plots"))
    
    # Get quaternion inverse
    q_inv = Quaternion.inverse(q=q)
    
    # Get dot product of two quaternions [q_1 = q_inv * q = 0]
    # Obviously multiplying a quaternion by its inverse is pointless, but it shows a dot_product usage example
    q_1 = Quaternion.dot_product(q1=q, q2=q_inv)
    
    # The original Euler angle can be recovered [e1 = e]
    e1 = Quaternion.to_euler(q=q)

    
if __name__ == "__main__":
    main()
```

![alt text](https://github.com/AlexCrownshaw/n_Body_Solver/blob/master/n_body_solver/solutions/2n_1e3iter_100et_12-10-23_12-58-38/Plots/Rotation_Anim_12-10-23_12-59-25.gif " rotation_animation")
