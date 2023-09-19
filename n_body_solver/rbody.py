import numpy as np
import pandas as pd

from n_body_solver.body import Body
from n_body_solver.rk4 import RK4


class RBody(Body):

    def __init__(self, m: float, x: list, v: list = None, a: list = None, x_ang: list = None, v_ang: list = None,
                 i: list = None, m_unit: str = "kg", x_unit: str = "m", data: pd.DataFrame = None):
        """

        :param m:
        :param x:
        :param v:
        :param a:
        :param x_ang:
        :param v_ang:
        :param i:
        :param m_unit:
        :param x_unit:
        :param data:
        """

        super().__init__(m=m, x=x, v=v, a=a, m_unit=m_unit, x_unit=x_unit, data=data)

        self._q = np.zeros(3)

        if i is not None:
            self._i = i
        else:
            self._i = np.array([0.16667] * 3)

        if x_ang is not None:
            self._x_ang = np.array(x_ang)
        else:
            self._x_ang = np.zeros(3)

            if v_ang is not None:
                self._v_ang = np.array(v_ang)
            else:
                self._v_ang = np.zeros(3)

        for header in ["psi", "theta", "phi", "psi_dot", "theta_dot", "phi_dot", "q_phi", "q_theta", "q_phi"]:
            self._data[header] = []

        self._rk4 = RK4(func=self._compute_state_derivative)

        self._dt: float = None

    @property
    def q(self) -> np.array:
        return self._q

    @q.setter
    def q(self, value: np.array) -> None:
        self._q = value

    @property
    def i(self) -> np.array:
        return self._i

    @i.setter
    def i(self, value: np.array) -> None:
        self._i = value

    @property
    def dt(self) -> np.array:
        return self._dt

    @dt.setter
    def dt(self, value: np.array) -> None:
        self._dt = value

    @property
    def x_ang(self) -> np.array:
        return self._x_ang

    @x_ang.setter
    def x_ang(self, value: np.array) -> None:
        self._x_ang = value

    @property
    def v_ang(self) -> np.array:
        return self._v_ang

    @v_ang.setter
    def v_ang(self, value: np.array) -> None:
        self._v_ang = value

    def store_state(self, i: int, t: float) -> None:
        """

        :return:
        """

        self._data.loc[len(self._data)] = [i, t,
                                           self._x[0], self._x[1], self._x[2],
                                           self._v[0], self._v[1], self._v[2],
                                           self._a[0], self._a[1], self._a[2],
                                           self._F_g[0], self._F_g[1], self._F_g[2],
                                           self._x_ang[0], self._x_ang[1], self._x_ang[2],
                                           self._v_ang[0], self.v_ang[1], self.v_ang[2],
                                           self._q[0], self._q[1], self._q[2]]

    def compute_rotation(self) -> np.array:
        state_vector = self.get_state_vector()

    def get_state_vector(self) -> np.array:
        pass

    def _compute_state_derivative(self, state_vector: np.array) -> np.array:
        return state_vector
