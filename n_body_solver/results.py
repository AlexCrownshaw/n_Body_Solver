import os
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from n_body_solver.body import Body
from n_body_solver.rbody import RBody
from n_body_solver import Constants


class Results:

    def __init__(self, bodies: list[np.array] = None, solution_path: str = None):
        """

        :param bodies:
        :param solution_path:
        """

        if bodies is not None:
            self._bodies: list[np.array] = bodies
            self._merge_data_blocks()
            self._solution_path: str = ""
            self._name: str = self.get_solution_name()
            self._solver_params: dict = self.get_solver_params()
        elif solution_path is not None:
            self.load_solution(solution_path=solution_path)
        else:
            raise Exception("ERROR: Results object created without valid bodies list or solution path")

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

    def _merge_data_blocks(self) -> None:
        for body in self._bodies:
            merged_data = pd.DataFrame()
            for data_block in body.data:
                merged_data = pd.concat([merged_data, data_block], axis=0)

            body.data = merged_data

    def get_solution_name(self) -> str:
        """

        :return:
        """

        iterations = round(max([len(body.data) for body in self._bodies]))
        et = max([max(body.data.time) for body in self._bodies])

        return f"{len(self._bodies)}n_{round(iterations / 1e3)}e3iter_{round(et)}et_{str(time.strftime('%d-%m-%y_%H-%M-%S'))}"

    def get_solver_params(self) -> dict:
        """

        :return:
        """

        body_params = [None] * len(self._bodies)
        for n, body in enumerate(self._bodies):
            body_params[n] = body.get_body_params()
            body_params[n]["n"] = n

        return {"iter": round(max([len(body.data) for body in self._bodies])),
                "et": max([max(body.data.time) for body in self._bodies]),
                "dt": round(self._bodies[0].data.time.iloc[1] - self._bodies[0].data.time.iloc[0], 3),
                "bodies": body_params}

    def save_solution(self, path: str = None) -> None:
        """

        :param path:
        :return:
        """

        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solutions", self._name)
        else:
            path = os.path.join(path, self._name)
        os.mkdir(path)

        for n, body in enumerate(self._bodies):
            body_path = os.path.join(path, f"n_{n}")
            if not os.path.isdir(body_path):
                os.mkdir(body_path)
            body.data.to_csv(os.path.join(body_path, f"n_{n}_data.csv"), index=False)

        solver_params = self.get_solver_params()
        with open(os.path.join(path, "Solver_Parameters.json"), "w") as json_file:
            json.dump(solver_params, json_file)

        self._solution_path = path

    def load_solution(self, solution_path: str) -> None:
        """

        :param solution_path:
        :return:
        """

        body_dir_names = [dir_name for dir_name in os.listdir(solution_path) if "n_" in dir_name]

        with open(os.path.join(solution_path, "Solver_Parameters.json"), "r") as json_file:
            self._solver_params = json.load(json_file)
            body_params = self._solver_params["bodies"]

        bodies = [None] * len(body_dir_names)
        for body_dir_name in body_dir_names:
            n = int(body_dir_name.replace("n_", ""))
            body_data = pd.read_csv(os.path.join(solution_path, body_dir_name, f"n_{n}_data.csv"))

            if body_params[n]["type"] == "body":
                bodies[n] = Body(m=body_params[n]["mass"], x=body_params[n]["x_init"], v=body_params[n]["v_init"], data=body_data)
            elif body_params[n]["type"] == "rbody":
                bodies[n] = RBody(m=body_params[n]["mass"], x=body_params[n]["x_init"], v=body_params[n]["v_init"],
                                  x_ang=body_params[n]["x_ang_init"], v_ang=body_params[n]["v_ang_init"], data=body_data)
            else:
                raise Exception(f"ERROR: body type {body_params[n]['type']} is not recognised by Results class")

        self._bodies = bodies
        self._solution_path = solution_path
        self._name = os.path.basename(solution_path)
        self._solver_params = self.get_solver_params()

    def recover_solver(self):
        """

        :return:
        """

        from n_body_solver.solver import Solver

        return Solver(bodies=self._bodies, iterations=self._solver_params["iter"], dt=self._solver_params["dt"])

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

    def animate_solution(self, frames: int = 20, n_filter: list[int] = None, show: bool = True,
                         save: bool = True) -> None:
        """

        :param frames:
        :param n_filter:
        :param show:
        :param save:
        :return:
        """

        if n_filter is None:
            n_filter = range(len(self._bodies))

        iter_step = round(max([len(body.data) for body in self._bodies]) / frames)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        frame_data = [ax.plot(self._bodies[n].data.x.iloc[0: iter_step],
                              self._bodies[n].data.y.iloc[0: iter_step],
                              self._bodies[n].data.z.iloc[0: iter_step],
                              markevery=[-1], marker="o",
                              label=f"n{n}_{round(self._bodies[n].m / Constants.MASS_UNITS['sm'])}[SM]")[0]
                      for n in range(len(self._bodies)) if n in n_filter]

        # max_disp = min_disp = float(0)
        # for body in self._bodies:
        #     max_disp_body = max([max(body.data[dim]) for dim in ["x", "y", "z"]])
        #     if max_disp_body > max_disp:
        #         max_disp = max_disp_body
        #
        #     min_disp_body = min([min(body.data[dim]) for dim in ["x", "y", "z"]])
        #     if min_disp_body > min_disp:

        ax.set(xlabel=f"X_[{'Km'}]")
        ax.set(ylabel=f"Y_[{'Km'}]")
        ax.set(zlabel=f"Z_[{'Km'}]")
        plt.legend()

        title = ""
        for part in [f"m{n}: {np.round(body.m/Constants.MASS_UNITS['sm'], 3)} [SM] --- x{n}_init: {np.round(body.x_init/Constants.DISP_UNITS['au'], 3)} [AU] --- v{n}_init: {np.round(body.v_init/Constants.VELOCITY_UNITS['kmps'], 3)} [Km/s]"
                     for n, body in enumerate(self._bodies) if n in n_filter]:
            title = f"{title}\n{part}"
        plt.title(title)

        anim = animation.FuncAnimation(fig=fig, func=self._update_frame_data, frames=frames,
                                       fargs=(frame_data, int(iter_step), n_filter), interval=50, blit=False)

        if save:
            plot_dir = os.path.join(self.solution_path, "Plots")
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            writer_gif = animation.PillowWriter(fps=60)
            file_path = os.path.join(self._solution_path, "Plots", f"Solution_Animation_{len(n_filter)}n.gif")
            anim.save(filename=file_path, writer=writer_gif)

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

        iter_index: int = num * iter_step

        xlim = np.zeros((2, len(n_filter)))
        ylim = np.zeros((2, len(n_filter)))
        zlim = np.zeros((2, len(n_filter)))

        for n, line in enumerate(frame_data):
            if n in n_filter:
                x_data = self._bodies[n].data.x.iloc[:iter_index]
                y_data = self._bodies[n].data.y.iloc[:iter_index]
                z_data = self._bodies[n].data.z.iloc[:iter_index]

                line.set_data(np.array([x_data, y_data]))
                line.set_3d_properties(zs=z_data)

                if num != 0:
                    xlim[:, n] = [max(x_data), min(x_data)]
                    ylim[:, n] = [max(y_data), min(y_data)]
                    zlim[:, n] = [max(z_data), min(z_data)]

        if num != 0:
            for line in frame_data:
                line.axes.set(xlim3d=(min(xlim[1, :]), max(xlim[0, :])),
                              ylim3d=(min(ylim[1, :]), max(ylim[0, :])),
                              zlim3d=(-1, 1))

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
            fig = plt.figure(figsize=(7, 7))

        for n in n_filter:
            v_data = np.array([self._bodies[n].data.v_x.iloc[iter_range[0]:iter_range[1]],
                               self._bodies[n].data.v_y.iloc[iter_range[0]:iter_range[1]],
                               self._bodies[n].data.v_z.iloc[iter_range[0]:iter_range[1]]])
            v_mag = [np.linalg.norm(v_data[:, i]) / Constants.VELOCITY_UNITS["kmps"] for i in range(v_data.shape[1])]
            plt.plot(self._bodies[n].data.time.iloc[iter_range[0]:iter_range[1]] / Constants.TIME_UNITS["year"], v_mag, label=f"n: {n}")

        plt.xlabel("Time [Years]")
        plt.ylabel("Velocity Magnitude [Km/s]")
        plt.legend()
        plt.grid()

        if save:
            self.save_plot(fig=fig, plot_name=f"Velocity_Mag_{len(n_filter)}n_{iter_range}iter_rng")

        if show:
            plt.show()

        return fig
