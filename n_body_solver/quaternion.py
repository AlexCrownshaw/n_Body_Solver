import numpy as np


class Quaternion:

    @staticmethod
    def euler_to_quaternion(e_vec: np.array) -> np.array:
        """

        :param e_vec: euler angle rotation vector
        return: quaternion rotation vector
        """

        e_vec = np.radians(e_vec / 2)
        cos_psi, cos_theta, cos_phi = np.cos(e_vec[0]), np.cos(e_vec[1]), np.cos(e_vec[2])
        sin_psi, sin_theta, sin_phi = np.sin(e_vec[0]), np.sin(e_vec[1]), np.sin(e_vec[2])

        q = np.zeros(4)
        q[0] = (cos_phi * cos_theta * cos_psi) + (sin_phi * sin_theta * sin_psi)
        q[1] = (sin_phi * cos_theta * cos_psi) - (cos_phi * sin_theta * sin_psi)
        q[2] = (cos_phi * sin_theta * cos_psi) + (sin_phi * cos_theta * sin_psi)
        q[3] = (cos_phi * cos_theta * sin_psi) - (sin_phi * sin_theta * cos_psi)

        return q

    @staticmethod
    def quaternion_to_euler(q: np.array) -> np.array:
        """

        :param q: quaternion matrix
        return: euler angle vector
        """

        e = np.zeros(3)
        e[0] = np.arctan2(2 * ((q[0] * q[3]) + (q[1] * q[2])), (q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2))
        e[1] = np.arcsin(2 * ((q[0] * q[2]) - (q[1] * q[3])))
        e[2] = np.arctan2(2 * ((q[0] * q[1]) + (q[2] * q[3])), (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2))

        return np.degrees(e)

    @staticmethod
    def quaternion_dot(q1: np.array, q2: np.array) -> np.array:
        """

        :param q1:
        :param q2:
        :return:
        """

        return np.array([q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
                         q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
                         q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
                         q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]])
