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

        self._q = self.euler_to_quaternion(e_vec=self._x_ang)

        for header in ["psi", "theta", "phi", "psi_dot", "theta_dot", "phi_dot", "T_psi", "T_theta", "T_phi"]:
            self._data[header] = []

        self._rk4 = RK4(func=self._compute_state_derivative)

        self._dt: float = None

    @property
    def q(self) -> np.array:
        return self._T

    @q.setter
    def q(self, value: np.array) -> None:
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

    def update_rotation_state(self, state_vector: np.array, T: np.array) -> None:
        """

        :param state_vector:
        :param T:
        :return:
        """

        self._x_ang = state_vector[:3]
        self._v_ang = state_vector[3:]
        self._T = T
        self._q = self.euler_to_quaternion(e_vec=self._x_ang)

    def store_state(self, i: int, t: float) -> None:
        """

        :param i:
        :param t:
        :return:
        """

        self._data.loc[len(self._data)] = [i, t,
                                           self._x[0], self._x[1], self._x[2],
                                           self._v[0], self._v[1], self._v[2],
                                           self._a[0], self._a[1], self._a[2],
                                           self._F_g[0], self._F_g[1], self._F_g[2],
                                           self._x_ang[0], self._x_ang[1], self._x_ang[2],
                                           self._v_ang[0], self.v_ang[1], self.v_ang[2],
                                           self._T[0], self._T[1], self._T[2]]

    @staticmethod
    def euler_to_quaternion(e_vec: np.array) -> np.array:
        """

        :param e_vec: euler angle rotation vector
        return: quaternion rotation vector
        """

        e_vec = np.radians(e_vec / 2)
        cos_psi, cos_theta, cos_phi = np.cos(e_vec[0]), np.cos(e_vec[1]), np.cos(e_vec[2])
        sin_psi, sin_theta, sin_phi = np.sin(e_vec[0]), np.sin(e_vec[1]), np.sin(e_vec[2])

        r = np.zeros(4)
        r[0] = (cos_phi * cos_theta * cos_psi) + (sin_phi * sin_theta * sin_psi)
        r[1] = (sin_phi * cos_theta * cos_psi) - (cos_phi * sin_theta * sin_psi)
        r[2] = (cos_phi * sin_theta * cos_psi) + (sin_phi * cos_theta * sin_psi)
        r[3] = (cos_phi * cos_theta * sin_psi) - (sin_phi * sin_theta * cos_psi)

        r_mag = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
        r_unit = r / r_mag

        return r_unit, r_mag

    @staticmethod
    def quaternion_to_euler(r: np.array) -> np.array:
        """

        :param r: quaternion matrix
        return: euler angle vector
        """

        e = np.zeros(3)
        e[0] = np.arctan2(2 * ((r[0] * r[3]) + (r[1] * r[2])), (r[0]**2 + r[1]**2 - r[2]**2 - r[3]**2))
        e[1] = np.arcsin(2 * ((r[0] * r[2]) - (r[1] * r[3])))
        e[2] = np.arctan2(2 * ((r[0] * r[1]) + (r[2] * r[3])), (r[0]**2 - r[1]**2 - r[2]**2 + r[3]**2))

        return np.degrees(e)

    @staticmethod
    def quaternion_dot(r1: np.array, r2: np.array) -> np.array:
        """

        :param r1:
        :param r2:
        :return:
        """

        return np.array([r1[0]*r2[0] - r1[1]*r2[1] - r1[2]*r2[2] - r1[3]*r2[3],
                         r1[0]*r2[1] + r1[1]*r2[0] + r1[2]*r2[3] - r1[3]*r2[2],
                         r1[0]*r2[2] - r1[1]*r2[3] + r1[2]*r2[0] + r1[3]*r2[1],
                         r1[0]*r2[3] + r1[1]*r2[2] - r1[2]*r2[1] + r1[3]*r2[0]])

    def compute_rotation(self, T: np.array) -> np.array:
        """

        :param T: Torque vector in the right-hand euler rotational coordinate system
        return:
        """

        if type(T) is list:
            T = np.array(T)

        state_vector = self.get_state_vector()
        state_vector = self._rk4.compute(dt=self._dt, state_vec=state_vector, args=[T])
        self.update_rotation_state(state_vector=state_vector, T=T)

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
