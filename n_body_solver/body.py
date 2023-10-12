import numpy as np
import pandas as pd

from typing import Union
from n_body_solver.constants import Constants


class Body:
    _DATA_BLOCK_SIZE = 1000
    _DATA_HEADERS = ["iteration", "time", "x", "y", "z", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "F_x", "F_y", "F_z"]

    def __init__(self, m: float, x: list, v: list = None, a: list = None, m_unit: str = "kg", x_unit: str = "m",
                 v_unit: str = "mps", data: pd.DataFrame = None):
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
        if m_unit in Constants.MASS_UNITS.keys():
            self._m_unit: str = m_unit
        else:
            raise Exception(f"ERROR: Invalid mass unit. Choose from {Constants.MASS_UNITS.keys()}")

        if x_unit in Constants.DISP_UNITS.keys():
            self._x_unit: str = x_unit
        else:
            raise Exception(f"ERROR: Invalid displacement unit. Choose from {Constants.DISP_UNITS.keys()}")

        if v_unit in Constants.VELOCITY_UNITS.keys():
            self._v_unit: str = v_unit
        else:
            raise Exception(f"ERROR: Invalid velocity unit. Choose from {Constants.VELOCITY_UNITS.keys()}")

        """ Scalar declarations """
        self._m: float = m * Constants.MASS_UNITS[self._m_unit]

        """ Cartesian vector declarations """
        self._F_g: np.array = np.zeros(3)
        self._x: np.array = np.array(x) * Constants.DISP_UNITS[self._x_unit]

        if v is None:
            self._v: np.array = np.zeros(3)
        else:
            self._v: np.array = np.array(v) * Constants.VELOCITY_UNITS[self._v_unit]

        if a is None:
            self._a: np.array = np.zeros(3)
        else:
            self._a: np.array = np.array(a) * Constants.DISP_UNITS[self._x_unit]

        """ State vector declaration """
        self._state_vec = np.array([self._x, self._v, self._a])

        """ Data frame declaration """
        if data is not None:
            self._data = [data]
        else:
            self._data = [self.create_data_block()]

        """ Store Initial Conditions """
        self._x_init: np.array = self._x
        self._v_init: np.array = self._v

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
    def data(self) -> Union[list, pd.DataFrame]:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if type(value) is not pd.DataFrame:
            raise Exception("ERROR: data.setter only allows use for combining data blocks for post processing")

        self._data = value

    @property
    def m_unit(self) -> str:
        return self._m_unit

    @property
    def x_unit(self) -> str:
        return self._x_unit

    @property
    def v_unit(self) -> str:
        return self._v_unit

    @property
    def x_init(self) -> np.array:
        return self._x_init

    @property
    def v_init(self) -> np.array:
        return self._v_init

    def create_data_block(self) -> pd.DataFrame:
        """

        :return:
        """

        return pd.DataFrame(np.zeros(shape=(self._DATA_BLOCK_SIZE, len(self._DATA_HEADERS))),
                            columns=self._DATA_HEADERS)

    def store_state(self, i: int, t: float) -> None:
        """

        :return:
        """

        index = i % self._DATA_BLOCK_SIZE

        if index == 0 and i != 0:
            self._data.append(self.create_data_block())

        self._data[-1].loc[index] = [i, t] + self.get_state_data()

    def get_state_data(self) -> list:
        """

        :return:
        """

        return [self._x[0], self._x[1], self._x[2],
                self._v[0], self._v[1], self._v[2],
                self._a[0], self._a[1], self._a[2],
                self._F_g[0], self._F_g[1], self._F_g[2]]

    def get_body_params(self) -> dict:
        """

        :return:
        """

        return {"n": 0, "type": "body", "mass": self.m, "x_init": [float(self.x_init[i]) for i in range(3)],
                "v_init": [float(self.v_init[i]) for i in range(3)]}
