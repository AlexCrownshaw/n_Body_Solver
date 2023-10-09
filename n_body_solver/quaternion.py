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

        r = np.zeros(4)
        r[0] = (cos_phi * cos_theta * cos_psi) + (sin_phi * sin_theta * sin_psi)
        r[1] = (sin_phi * cos_theta * cos_psi) - (cos_phi * sin_theta * sin_psi)
        r[2] = (cos_phi * sin_theta * cos_psi) + (sin_phi * cos_theta * sin_psi)
        r[3] = (cos_phi * cos_theta * sin_psi) - (sin_phi * sin_theta * cos_psi)

        r_mag = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
        r_unit = r / r_mag

        return r_unit, r_mag

    @staticmethod
    def quaternion_to_euler(r: np.array) -> np.array:
        """

        :param r: quaternion matrix
        return: euler angle vector
        """

        e = np.zeros(3)
        e[0] = np.arctan2(2 * ((r[0] * r[3]) + (r[1] * r[2])), (r[0]**2 + r[1]**2 - r[2]**2 - r[3]**2))
        e[1] = np.arcsin(2 * ((r[0] * r[2]) - (r[1] * r[3])))
        e[2] = np.arctan2(2 * ((r[0] * r[1]) + (r[2] * r[3])), (r[0]**2 - r[1]**2 - r[2]**2 + r[3]**2))

        return np.degrees(e)

    @staticmethod
    def quaternion_dot(r1: np.array, r2: np.array) -> np.array:
        """

        :param r1:
        :param r2:
        :return:
        """

        return np.array([r1[0]*r2[0] - r1[1]*r2[1] - r1[2]*r2[2] - r1[3]*r2[3],
                         r1[0]*r2[1] + r1[1]*r2[0] + r1[2]*r2[3] - r1[3]*r2[2],
                         r1[0]*r2[2] - r1[1]*r2[3] + r1[2]*r2[0] + r1[3]*r2[1],
                         r1[0]*r2[3] + r1[1]*r2[2] - r1[2]*r2[1] + r1[3]*r2[0]])
