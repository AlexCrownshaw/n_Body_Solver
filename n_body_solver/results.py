import os
import time

import numpy as np
import matplotlib.pyplot as plt


class Results:

    def __init__(self, bodies: list[np.array]):
        """

        :param bodies:
        """

        self._bodies: list[np.array] = bodies
        self._name: str = str(time.strftime("%d-%m-%y %H-%M-%S"))

    @property
    def bodies(self) -> list[np.array]:
        return self._bodies

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def save_plot(self, fig) -> None:
        """

        :param fig:
        :return:
        """

        pass

    def plot_trajectory(self, n_filter: list[int] = None, fig: plt.Figure = None, show: bool = True, save: bool = True) -> plt.Figure:
        """

        :param n_filter:
        :param fig:
        :param show:
        :param save:
        :return:
        """

        if n_filter is None:
            n_filter = range(len(self._bodies))

        if fig is None:
            fig = plt.Figure(figsize=(10, 10))
            ax = plt.axes(projection="3d")
        else:
            ax = fig.axes[0]

        for n in n_filter:
            lines = ax.plot3D(self._bodies[n].data.x, self._bodies[n].data.y, self._bodies[n].data.z, label=f"n: {n}")
            ax.plot(self._bodies[n].data.x.iloc[-1], self._bodies[n].data.y.iloc[-1], self._bodies[n].data.z.iloc[-1], marker="o", color=lines[-1].get_color())
            ax.plot(self._bodies[n].data.x.iloc[0], self._bodies[n].data.y.iloc[0], self._bodies[n].data.z.iloc[0], marker="x", color=lines[-1].get_color())

        ax.set_xlabel("x [Km]")
        ax.set_ylabel("y [Km]")
        ax.set_zlabel("z [Km]")
        plt.legend()

        if show:
            plt.show()

        if save:
            self.save_plot(fig=fig)

        return fig

    def save_solution(self, path: str = None) -> None:
        """

        :param path:
        :return:
        """

        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions", self._name)
            os.mkdir(path)

        for n, body in enumerate(self._bodies):
            body_path = os.path.join(path, f"n_{n}")
            if not os.path.isdir(body_path):
                os.mkdir(body_path)

            body.data.to_csv(os.path.join(body_path, f"n_{n}_data.csv"), index=False)
