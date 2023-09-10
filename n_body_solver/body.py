import numpy as np
import pandas as pd


class Body:

    _MASS_UNITS = {"kg": 1,
                   "sm": 1.988e30}
    _DISP_UNITS = {"m": 1,
                   "au": 1.496e+11}

    def __init__(self, m: float, x: list, v: list = None, a: list = None, m_unit: str = "kg", x_unit: str = "m",
                 data: pd.DataFrame = None):
        """

        :param m:
        :param x:
        :param v:
        :param a:
        :param m_unit:
        :param x_unit:
        :param data:
        """

        """ Unit declarations """
        if m_unit in self._MASS_UNITS.keys():
            self._m_unit: str = m_unit
        else:
            raise Exception(f"ERROR: Invalid mass unit. Choose from {self._MASS_UNITS.keys()}")

        if x_unit in self._DISP_UNITS.keys():
            self._x_unit: str = x_unit
        else:
            raise Exception(f"ERROR: Invalid displacement unit. Choose from {self._DISP_UNITS.keys()}")

        """ Scalar declarations """
        self._m: float = self.convert_to_kg(mass=m, unit=self._m_unit)

        """ Cartesian vector declarations """
        self._F_g: np.array = np.zeros(3)
        self._x: np.array = self.convert_to_meters(vec=np.array(x), unit=self.x_unit)

        if v is None:
            self._v = np.zeros(3)
        else:
            self._v = self.convert_to_meters(vec=np.array(v), unit=self.x_unit)

        if a is None:
            self._a = np.zeros(3)
        else:
            self._a = self.convert_to_meters(vec=np.array(a), unit=self.x_unit)

        """ State vector declaration """
        self._state_vec = np.array([self._x, self._v, self._a])

        """ Data frame declaration """
        if data is not None:
            self._data = data
        else:
            self._data = pd.DataFrame(columns=["iteration", "time",
                                               "x", "y", "z",
                                               "v_x", "v_y", "v_z",
                                               "a_x", "a_y", "a_z",
                                               "F_x", "F_y", "F_z"])

        """ Store Initial Conditions """
        self._x_ic: np.array = self._x
        self._v_ic: np.array = self._v

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

    @property
    def m_unit(self) -> str:
        return self._m_unit

    @property
    def x_unit(self) -> str:
        return self._x_unit

    @property
    def x_ic(self) -> np.array:
        return self._x_ic

    @property
    def v_ic(self) -> np.array:
        return self._v_ic

    def store_state(self, i: int, t: float) -> None:
        """

        :return:
        """

        self._data.loc[len(self._data)] = [i, t,
                                           self._x[0], self._x[1], self._x[2],
                                           self._v[0], self._v[1], self._v[2],
                                           self._a[0], self._a[1], self._a[2],
                                           self._F_g[0], self._F_g[1], self._F_g[2]]

    def convert_to_kg(self, mass: float, unit: str) -> float:
        """

        :param mass:
        :param unit:
        :return:
        """

        return mass * self._MASS_UNITS[unit]

    def convert_to_meters(self, vec: np.array, unit: str) -> np.array:
        """

        :param vec:
        :param unit:
        :return:
        """

        return vec * self._DISP_UNITS[unit]
