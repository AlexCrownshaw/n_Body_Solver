import numpy as np

from n_body_solver.body import Body
from n_body_solver.rbody import RBody
from n_body_solver.rk4 import RK4
from n_body_solver.results import Results
from n_body_solver.constants import Constants


class Solver:

    def __init__(self, bodies: list[Body], iterations: float = 100, dt: float = 1, debug=True):
        """

        :param bodies:
        :param iterations:
        :param dt:
        :param debug:
        """

        self._debug: bool = debug
        self._bodies: list[np.array] = bodies
        self._rk4 = RK4(func=self._compute_state_derivative)

        self._iterations: float = iterations
        self._dt: float = dt
        self._t: float = 0

        for body in self._bodies:
            if type(body) is RBody:
                body.dt = self._dt
                body.dt = self._dt

    @property
    def bodies(self) -> list[Body]:
        return self._bodies

    @property
    def iterations(self) -> float:
        return self._iterations

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def t(self) -> float:
        return self._t

    def config_solver(self, iterations: float, dt: float) -> None:
        """

        :param iterations:
        :param dt:
        :return:
        """

        self._iterations = iterations
        self._dt = dt

    def _compute_F_g(self, n_target: int) -> np.array:
        """

        :param n_target:
        :return:
        """

        F_g = np.zeros(3)

        for n, body in enumerate(self._bodies):
            if n != n_target:
                x_rel = body.x - self._bodies[n_target].x
                F_g += (Constants.G * self._bodies[n_target].m * self._bodies[n].m * x_rel) / (
                            np.linalg.norm(x_rel, ord=1) ** 3)

        return F_g

    def _compute_state_derivative(self, state_vec: np.array) -> np.array:
        """

        :param state_vec:
        :return:
        """

        state_derivative = np.zeros((len(self._bodies), 6))
        for n, _ in enumerate(self._bodies):
            state_derivative[n, :3] = state_vec[n, 3:]
            state_derivative[n, 3:] = self._compute_F_g(n_target=n) / self._bodies[n].m

        return state_derivative

    def compute_iteration(self) -> np.array:
        """

        :return:
        """

        state_vec = np.array([self._get_state_vector(n=n) for n in range(len(self._bodies))])
        state_vec = self._rk4.compute(dt=self._dt, state_vec=state_vec)

        return state_vec

    def _get_state_vector(self, n) -> np.array:
        """

        :param n:
        :return:
        """

        vec = np.zeros(6)
        vec[:3] = self._bodies[n].x
        vec[3:] = self._bodies[n].v

        return vec

    def _update_bodies(self, state_vec: np.array, i: int, t: float) -> None:
        """

        :param state_vec:
        :return:
        """

        for n, body in enumerate(self._bodies):
            body.x = state_vec[n, :3]
            body.v = state_vec[n, 3:]
            body.a = self._compute_F_g(n_target=n) / self._bodies[n].m
            body.store_state(i=i, t=t)

    def init_solver(self) -> None:
        """
        Generates the zeroth iteration data
        :return:
        """

        state_vec = np.array([self._get_state_vector(n=n) for n in range(len(self._bodies))])
        self._update_bodies(state_vec=state_vec, i=0, t=self._t)
        self.print_debug(i=0)

    def solve(self) -> Results:
        """

        :return:
        """

        self.init_solver()

        for i in np.arange(1, self._iterations):
            self._t = i * self._dt

            # if i == 0:
            #     state_vec = np.array([self._get_state_vector(n=n) for n in range(len(self._bodies))])
            # else:
            #     state_vec = self._compute_iteration()

            state_vec = self.compute_iteration()

            for body in self._bodies:
                if type(body) is RBody:
                    body.compute_rotation()

            self._update_bodies(state_vec=state_vec, i=i, t=self._t)

            self.print_debug(i=i)

        return Results(bodies=self._bodies)

    def print_debug(self, i) -> None:
        """

        :return:
        """

        if self._debug:
            print(f"\nInstance: {i}, Time: {self._t} s")
            for n, body in enumerate(self._bodies):
                print(f"""n: {n}
                             \tx: {np.round(body.x)}
                             \tv: {np.round(body.v)}
                             \ta: {np.round(body.a)}
                             \tF_g: {np.round(body.F_g)}""")
