import numpy as np
import pandas as pd


class Body:

    def __init__(self, m: float, x: list, v: list = None, a: list = None):
        """

        :param m:
        :param x:
        """

        """ Scalar declarations """
        self._m: float = m

        """ Cartesian vector declarations """
        self._F_g: np.array = np.zeros(3)
        self._x: np.array = np.array(x)

        if v is None:
            self._v = np.zeros(3)
        else:
            self._v = np.array(v)

        if a is None:
            self._a = np.zeros(3)
        else:
            self._a = np.array(a)

        """ State vector declaration """
        self._state_vec = np.array([self._x, self._v, self._a])

        self._data = pd.DataFrame(columns=["iteration", "time",
                                           "x", "y", "z",
                                           "v_x", "v_y", "v_z",
                                           "a_x", "a_y", "a_z",
                                           "F_x", "F_y", "F_z"])

    @property
    def m(self) -> float:
        return self._m

    @property
    def F_g(self) -> np.array:
        return self._F_g

    @F_g.setter
    def F_g(self, value: np.array) -> None:
        self._F_g = value.reshape(3)

    @property
    def x(self) -> np.array:
        return self._x

    @x.setter
    def x(self, value: np.array) -> None:
        self._x = value.reshape(3)

    @property
    def v(self) -> np.array:
        return self._v

    @v.setter
    def v(self, value: np.array) -> None:
        self._v = value.reshape(3)

    @property
    def a(self) -> np.array:
        return self._a

    @a.setter
    def a(self, value: np.array) -> None:
        self._a = value.reshape(3)

    @property
    def state_vec(self) -> np.array:
        return self._state_vec

    @state_vec.setter
    def state_vec(self, value: np.array) -> None:
        self._state_vec = value

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def store_state(self, i: int, t: float) -> None:
        """

        :return:
        """

        self._data.loc[len(self._data)] = [i, t,
                                           self._x[0], self._x[1], self._x[2],
                                           self._v[0], self._v[1], self._v[2],
                                           self._a[0], self._a[1], self._a[2],
                                           self._F_g[0], self._F_g[1], self._F_g[2]]
