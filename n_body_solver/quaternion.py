import numpy as np
import matplotlib.pyplot as plt


class Quaternion:

    @staticmethod
    def from_euler(e: np.array) -> np.array:
        """
        Returns a quaternion matrix from a right hand Tait-Bryan Euler angle matrix
        :param e: right hand Tait-Bryan Euler angle matrix
        return: quaternion matrix
        """

        if type(e) is list:
            e = np.array(e)

        e = np.radians(e / 2)
        cos_psi, cos_theta, cos_phi = np.cos(e[0]), np.cos(e[1]), np.cos(e[2])
        sin_psi, sin_theta, sin_phi = np.sin(e[0]), np.sin(e[1]), np.sin(e[2])

        q = np.zeros(4)
        q[0] = (cos_phi * cos_theta * cos_psi) + (sin_phi * sin_theta * sin_psi)
        q[1] = (sin_phi * cos_theta * cos_psi) - (cos_phi * sin_theta * sin_psi)
        q[2] = (cos_phi * sin_theta * cos_psi) + (sin_phi * cos_theta * sin_psi)
        q[3] = (cos_phi * cos_theta * sin_psi) - (sin_phi * sin_theta * cos_psi)

        return q

    @staticmethod
    def to_euler(q: np.array) -> np.array:
        """
        Returns a right hand Tait-Bryan Euler angle matrix from a quaternion matrix
        :param q: quaternion matrix
        return: euler angle vector
        """

        e = np.zeros(3)
        e[0] = np.arctan2(2 * ((q[0] * q[3]) + (q[1] * q[2])), (q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2))
        e[1] = np.arcsin(2 * ((q[0] * q[2]) - (q[1] * q[3])))
        e[2] = np.arctan2(2 * ((q[0] * q[1]) + (q[2] * q[3])), (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2))

        return np.degrees(e)

    @staticmethod
    def dot_product(q1: np.array, q2: np.array) -> np.array:
        """
        Quaternion dot product multiplication (Non-commutable)
        :param q1:
        :param q2:
        :return: q2 * q1
        """

        return np.array([q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                         q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                         q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                         q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])

    @staticmethod
    def inverse(q: np.array) -> np.array:
        """

        :param q:
        :return:
        """

        return np.array([q[0], -q[1], -q[2], -q[3]])

    @classmethod
    def rotate_point(cls, p: np.array, q: np.array, active: bool = True) -> np.array:
        """

        :param p:
        :param q:
        :param active:
        :return:
        """

        p = np.array([0, p[0], p[1], p[2]])
        q_inv = cls.inverse(q=q)

        if active:
            p_prime = cls.dot_product(q1=cls.dot_product(q1=q_inv, q2=p), q2=q)
        else:
            p_prime = cls.dot_product(q1=cls.dot_product(q1=q, q2=p), q2=q_inv)

        return p_prime[1:]

    @classmethod
    def plot_quaternion(cls, q: list) -> None:
        """

        :param q:
        :return:
        """

        p_x = cls.rotate_point(p=[1, 0, 0], q=q, active=True)
        p_y = cls.rotate_point(p=[0, 1, 0], q=q, active=True)
        p_z = cls.rotate_point(p=[0, 0, -1], q=q, active=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.quiver(0, 0, 0, p_x[0], p_x[1], p_x[2], length=1, color="r", label="x")
        ax.quiver(0, 0, 0, p_y[0], p_y[1], p_y[2], length=1, color="g", label="y")
        ax.quiver(0, 0, 0, p_z[0], p_z[1], p_z[2], length=1, color="b", label="z")

        ax.plot([0, 1], [0, 0], [0, 0], color="r", linestyle="--")
        ax.plot([0, 0], [0, 1], [0, 0], color="g", linestyle="--")
        ax.plot([0, 0], [0, 0], [0, 1], color="b", linestyle="--")

        ax.set(xlim=(-1, 1))
        ax.set(ylim=(-1, 1))
        ax.set(zlim=(-1, 1))

        ax.set(xticks=[-1, 0, 1])
        ax.set(yticks=[-1, 0, 1])
        ax.set(zticks=[-1, 0, 1])

        plt.legend()

        plt.show()
