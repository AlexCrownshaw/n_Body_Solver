import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


class Results:

    def __init__(self, bodies: list[np.array]):
        """

        :param bodies:
        """

        self._bodies: list[np.array] = bodies
        self._name: str = self.get_solution_name()
        self._solution_path: str = ""

    @property
    def bodies(self) -> list[np.array]:
        return self._bodies

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def solution_path(self) -> str:
        return self._solution_path

    @solution_path.setter
    def solution_path(self, value: str) -> None:
        self._solution_path = value

    def get_solution_name(self) -> str:
        """

        :return:
        """

        iterations = round(max([len(body.data) for body in self._bodies]))
        et = max([max(body.data.time) for body in self._bodies])

        return f"{len(self._bodies)}n_{round(iterations/1e3)}e3iter_{round(et)}et_{str(time.strftime('%d-%m-%y_%H-%M-%S'))}"

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

        self._solution_path = path

    def save_plot(self, fig, plot_name: str) -> None:
        """

        :param plot_name:
        :param fig:
        :return:
        """

        if self._solution_path is None:
            self.save_solution()

        plot_path = os.path.join(self._solution_path, "Plots")
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        # plt.figure(fig.number)

        plt.savefig(os.path.join(plot_path, f"{plot_name}.png"))

    def plot_trajectory(self, n_filter: list[int] = None, iter_range: list = None, fig: plt.Figure = None,
                        show: bool = True, save: bool = True) -> plt.Figure:
        """

        :param n_filter:
        :param iter_range:
        :param fig:
        :param show:
        :param save:
        :return:
        """

        if iter_range is None:
            iter_range = [0, max([len(body.data) for body in self._bodies])]

        if n_filter is None:
            n_filter = range(len(self._bodies))

        if fig is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection="3d")
        else:
            ax = fig.axes[0]

        for n in n_filter:
            lines = ax.plot3D(self._bodies[n].data.x.iloc[iter_range[0]: iter_range[1]],
                              self._bodies[n].data.y.iloc[iter_range[0]: iter_range[1]],
                              self._bodies[n].data.z.iloc[iter_range[0]: iter_range[1]], label=f"n: {n}")
            ax.plot3D(self._bodies[n].data.x.iloc[-1],
                      self._bodies[n].data.y.iloc[-1],
                      self._bodies[n].data.z.iloc[-1], marker="o", color=lines[-1].get_color())
            ax.plot3D(self._bodies[n].data.x.iloc[0],
                      self._bodies[n].data.y.iloc[0],
                      self._bodies[n].data.z.iloc[0], marker="x", color=lines[-1].get_color())

        ax.set_xlabel("x [Km]")
        ax.set_ylabel("y [Km]")
        ax.set_zlabel("z [Km]")
        plt.legend()

        if save:
            self.save_plot(fig=fig, plot_name=f"Trajectory_{len(n_filter)}n_{iter_range}iter_rng")

        if show:
            plt.show()

        return fig

    def animate_solution(self, frames: int = 20, n_filter: list[int] = None, show: bool = True) -> None:
        """

        :param frames:
        :param n_filter:
        :param show:
        :return:
        """

        if n_filter is None:
            n_filter = range(len(self._bodies))

        iter_step = round(max([len(body.data) for body in self._bodies]) / frames)

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        frame_data = [ax.plot(self._bodies[n].data.x.iloc[0],
                              self._bodies[n].data.x.iloc[0],
                              self._bodies[n].data.x.iloc[0])[0]
                      for n in range(len(self._bodies)) if n in n_filter]

        _ = animation.FuncAnimation(fig=fig, func=self._update_frame_data, frames=frames,
                                    fargs=(frame_data, int(iter_step), n_filter), interval=50, blit=False)

        if show:
            plt.show()

    def _update_frame_data(self, num: int, frame_data: np.array, iter_step: int,
                           n_filter: list[int] = None) -> np.array:
        """

        :param num:
        :param frame_data:
        :param iter_step:
        :param n_filter:
        :return:
        """

        if n_filter is None:
            n_filter = range(len(self._bodies))

        # iter_range = [iter_step * num, (iter_step * num) + iter_step]

        for n, ax in enumerate(frame_data):
            if n in n_filter:
                ax.set_data([self._bodies[n].data.x.iloc[iter_step * num],
                             self._bodies[n].data.y.iloc[iter_step * num]])
                ax.set_3d_properties([self._bodies[n].data.z.iloc[iter_step * num]])

        return frame_data

    def plot_velocity(self, n_filter: list[int] = None, iter_range: list = None, fig: plt.Figure = None,
                      show: bool = True, save: bool = True) -> plt.Figure:
        """

        :param n_filter:
        :param iter_range:
        :param fig:
        :param show:
        :param save:
        :return:
        """

        if iter_range is None:
            iter_range = [0, max([len(body.data) for body in self._bodies])]

        if n_filter is None:
            n_filter = range(len(self._bodies))

        if fig is None:
            fig = plt.figure(figsize=(10, 10))

        for n in n_filter:
            v_data = np.array([self._bodies[n].data.v_x.iloc[iter_range[0]:iter_range[1]],
                               self._bodies[n].data.v_y.iloc[iter_range[0]:iter_range[1]],
                               self._bodies[n].data.v_z.iloc[iter_range[0]:iter_range[1]]])
            v_mag = [np.linalg.norm(v_data[:, i]) for i in range(v_data.shape[1])]
            plt.plot(self._bodies[n].data.time.iloc[iter_range[0]:iter_range[1]], v_mag, label=f"n: {n}")

        plt.xlabel("Time [s]")
        plt.ylabel("Velocity Magnitude [m/s]")
        plt.legend()
        plt.grid()

        if save:
            self.save_plot(fig=fig, plot_name=f"Velocity_Mag_{len(n_filter)}n_{iter_range}iter_rng")

        if show:
            plt.show()
