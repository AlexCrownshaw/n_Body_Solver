import numpy as np

from typing import Callable


class RK4:

    def __init__(self, func: Callable):
        """

        :param func: Function used to compute the state derivative given the current state vector and a list of args
        """

        self._func = func

    @property
    def func(self) -> Callable:
        return self._func

    def compute(self, dt: float, state_vec: np.array, args: list = None) -> np.array:
        """

        :param dt:
        :param state_vec:
        :param args:
        :return:
        """

        if args is None:
            args = []

        k1 = self._func(state_vec, *args)
        k2 = self._func(state_vec + k1 * dt/2, *args)
        k3 = self._func(state_vec + k2 * dt/2, *args)
        k4 = self._func(state_vec + k3 * dt, *args)

        return state_vec + (1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)) * dt
