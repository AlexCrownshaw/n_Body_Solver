import numpy as np
import pandas as pd

from n_body_solver.body import Body
from n_body_solver.rk4 import RK4
from n_body_solver.quaternion import Quaternion


class RBody(Body):

    _DATA_HEADERS = ["psi", "theta", "phi", "psi_dot", "theta_dot", "phi_dot", "T_psi", "T_theta", "T_phi",
                     "q_w", "q_x", "q_y", "q_z"]

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

        self._DATA_HEADERS = super()._DATA_HEADERS + self._DATA_HEADERS

        super().__init__(m=m, x=x, v=v, a=a, m_unit=m_unit, x_unit=x_unit, data=data)

        self._T = np.zeros(3)

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

        self._x_ang_init = self._x_ang
        self._v_ang_init = self._v_ang

        self._q = Quaternion.from_euler(e=self._x_ang)

        self._rk4 = RK4(func=self._compute_state_derivative)

        self._dt: float = None

    @property
    def T(self) -> np.array:
        return self._T

    @T.setter
    def T(self, value: np.array) -> None:
        self._T = value

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

    @property
    def x_ang_init(self) -> np.array:
        return self._x_ang_init

    @property
    def v_ang_init(self) -> np.array:
        return self._v_ang_init

    def get_state_data(self) -> list:
        """

        :return:
        """

        state_data = super().get_state_data()

        return state_data + [self._x_ang[0], self._x_ang[1], self._x_ang[2],
                             self._v_ang[0], self.v_ang[1], self.v_ang[2],
                             self._T[0], self._T[1], self._T[2],
                             self._q[0], self._q[1], self._q[2], self._q[3]]

    def get_quaternion_data(self, iter_range: list = None) -> np.array:
        """

        :param iter_range:
        :return:
        """

        if iter_range is None:
            iter_range = [0, len(self._data.iteration)]

        q_data = np.zeros(shape=(iter_range[1], 4))
        for index in range(iter_range[0], iter_range[1]):
            q_data[index, :] = [self._data.q_w.iloc[index], self._data.q_x.iloc[index], self._data.q_y.iloc[index], self._data.q_z.iloc[index]]

        return q_data

    def get_body_params(self) -> dict:
        """

        :return:
        """

        return {"n": 0, "type": "rbody", "mass": self.m, "x_init": [float(self._x_init[i]) for i in range(3)],
                "v_init": [float(self._v_init[i]) for i in range(3)],
                "x_ang_init": [float(self._x_ang_init[i]) for i in range(3)],
                "v_ang_init": [float(self._v_ang_init[i]) for i in range(3)]}

    def _update_body_rotation(self, state_vector: np.array, T: np.array) -> None:
        """

        :param state_vector:
        :param T:
        :return:
        """

        self._T = T

        q_vec = Quaternion.from_euler(e=state_vector[:3])
        self._q = Quaternion.dot_product(q1=q_vec, q2=self._q, norm=True)

        self._x_ang = Quaternion.to_euler(q=self._q)
        self._v_ang = state_vector[3:]

    def compute_rotation(self, T: np.array) -> np.array:
        """

        :param T: Torque vector in the right-hand euler rotational coordinate system
        return:
        """

        if type(T) is list:
            T = np.array(T)

        state_vector = self.get_state_vector()
        state_vector = self._rk4.compute(dt=self._dt, state_vec=state_vector, args=[T])
        self._update_body_rotation(state_vector=state_vector, T=T)

    def get_state_vector(self) -> np.array:
        """

        return:
        """

        vec = np.zeros(6)
        vec[3:] = self._v_ang

        return vec

    def _compute_state_derivative(self, state_vector: np.array, T: np.array) -> np.array:
        """

        :param state_vector:
        :param T:
        :return:
        """

        vec = np.zeros(6)
        vec[:3] = state_vector[3:]
        vec[3:] = T / self._i

        return vec
