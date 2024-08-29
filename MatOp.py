import numpy as np


class MatOp:

    @staticmethod
    def zyx_rotation_matrix(cls, alpha, beta, gamma):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        cg = np.cos(gamma)
        sg = np.sin(gamma)
        return np.array(
            [
                [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                [-sb, cb * sg, cb * cg],
            ]
        )

    @staticmethod
    def dh_transform(a_i_minus_1, alpha_i_minus_1, d_i, theta_i):
        c_theta = np.cos(theta_i)
        s_theta = np.sin(theta_i)
        c_alpha = np.cos(alpha_i_minus_1)
        s_alpha = np.sin(alpha_i_minus_1)
        return np.array(
            [
                [c_theta, -s_theta, 0, a_i_minus_1],
                [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * d_i],
                [s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * d_i],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def dh_matrices(a_i_minus_1, alpha_i_minus_1, d_i, theta_i) -> list:
        N_FRAMES = len(a_i_minus_1)
        dh_matrices = []  # a list of transforms that describe frame {i} wrt frame {i-1}
        for i in range(N_FRAMES):
            dh_matrices.append(
                MatOp.dh_transform(
                    a_i_minus_1[i], alpha_i_minus_1[i], d_i[i], theta_i[i]
                )
            )
        return dh_matrices

    @staticmethod
    def inverse_transform(T):
        T_inv = np.eye(4)
        R = T[0:3, 0:3]
        P = T[0:3, 3]
        T_inv[0:3, 0:3] = R.T
        T_inv[0:3, 3] = -R.T @ P
        return T_inv

    @staticmethod
    def wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi