import numpy as np
from MatOp import MatOp

# define constants
PI_OVER_2 = np.pi / 2


class IkSolver:
    def __init__(self, dh_table) -> None:
        self.a_i_minus_1 = dh_table[0]
        self.alpha_i_minus_1 = dh_table[1]
        self.d_i = dh_table[2]

    def __get_wrist_from_tooltip(
        self, P: np.ndarray, R: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        T_0T = np.eye(4)
        T_0T[0:3, 0:3] = R
        T_0T[0:3, 3] = P

        T_6T = np.eye(4)
        T_6T[2, 3] = self.d_i[6]
        T_T6 = MatOp.inverse_transform(T_6T)
        T_06 = T_0T @ T_T6

        P = T_06[0:3, 3]
        R = T_06[0:3, 0:3]
        return P, R

    def __solve_theta3(self, P: np.ndarray, R: np.ndarray) -> float:
        # rename variables for readability
        a1 = self.a_i_minus_1[1]
        a2 = self.a_i_minus_1[2]
        a3 = self.a_i_minus_1[3]
        d1 = self.d_i[0]
        d4 = self.d_i[3]
        d_tool = self.d_i[-1]
        r13 = R[0, 2]
        r33 = R[2, 2]
        px = P[0]
        pz = P[2]

        # taken from matlab output
        theta3 = np.atan2(
            (
                a2 * d4
                - a1 * a3
                + d1 * d4
                + a3 * px
                - d4 * pz
                - a3 * d_tool * r13
                + d4 * d_tool * r33
            )
            / (
                a1**2
                + 2 * a1 * d_tool * r13
                - 2 * a1 * px
                + a2**2
                + 2 * a2 * d1
                + 2 * a2 * d_tool * r33
                - 2 * a2 * pz
                + d1**2
                + 2 * d1 * d_tool * r33
                - 2 * d1 * pz
                + d_tool**2 * r13**2
                + d_tool**2 * r33**2
                - 2 * d_tool * px * r13
                - 2 * d_tool * pz * r33
                + px**2
                + pz**2
            ),
            -(
                a2 * a3
                + a3 * d1
                + a1 * d4
                - a3 * pz
                - d4 * px
                + a3 * d_tool * r33
                + d4 * d_tool * r13
            )
            / (
                a1**2
                + 2 * a1 * d_tool * r13
                - 2 * a1 * px
                + a2**2
                + 2 * a2 * d1
                + 2 * a2 * d_tool * r33
                - 2 * a2 * pz
                + d1**2
                + 2 * d1 * d_tool * r33
                - 2 * d1 * pz
                + d_tool**2 * r13**2
                + d_tool**2 * r33**2
                - 2 * d_tool * px * r13
                - 2 * d_tool * pz * r33
                + px**2
                + pz**2
            ),
        )

        return theta3

    def __solve_theta_4(
        self, R: np.ndarray, theta3: np.ndarray, theta5: float
    ) -> float:
        # rename variables for readability
        s5 = np.sin(theta5)
        c3 = np.cos(theta3)
        s3 = np.sin(theta3)
        r_33 = R[2, 2]
        r_13 = R[0, 2]
        r_23 = R[1, 2]

        # solve for theta_4
        theta4 = np.atan2(-(r_33 * c3 + r_13 * s3) / s5, r_23 / s5)
        return theta4

    def __solve_theta_5(self, R: np.ndarray, theta3: float) -> tuple[float]:
        # rename variables for readability
        r13 = R[0, 2]
        r33 = R[2, 2]
        r23 = R[1, 2]
        c3 = np.cos(theta3)
        s3 = np.sin(theta3)

        # solve for theta_5
        theta5 = np.atan2(np.sqrt((r33*c3 + r13*s3)**2 + r23**2), r13*c3 - r33*s3)
        return theta5

    def __solve_theta_6(self, R: np.ndarray, theta3: float, theta5: float) -> float:
        # rename variables for readability
        r_12 = R[0, 1]
        r_32 = R[2, 1]
        r_11 = R[0, 0]
        r_31 = R[2, 0]
        s5 = np.sin(theta5)
        c3 = np.cos(theta3)
        s3 = np.sin(theta3)

        # solve for theta_6
        theta6 = np.atan2(-(r_12 * c3 - r_32 * s3) / s5, (r_11 * c3 - r_31 * s3) / s5)
        return theta6

    def solve_ik(self, *args) -> list:
        """
        Solves the inverse kinematics problem for a robotic arm.

        Args:
            *args: Variable number of arguments. The arguments can be provided in three different formats:
            - If len(args) == 6: The arguments are P_x, P_y, P_z, roll, pitch, and yaw angles.
            - If len(args) == 2: The arguments are P (position) and R (rotation matrix).
            - If len(args) == 1: The argument is a 4x4 transformation matrix.

        Returns:
            list: A list of joint angles theta 3 to 6 that solve the inverse kinematics problem.

        Raises:
            ValueError: If an invalid number of arguments is provided.
        """

        # ensure correct method usage
        if len(args) == 6:
            P = [args[0], args[1], args[2]]
            R = MatOp.zyx_rotation_matrix(args[3], args[4], args[5])
        elif len(args) == 2:
            P = args[0]
            R = args[1]
        elif len(args) == 1:
            P = args[0][0:3, 3]
            R = args[0][0:3, 0:3]
        else:
            raise ValueError("Invalid number of arguments provided.")

        P, R = self.__get_wrist_from_tooltip(P, R)
        theta3 = self.__solve_theta3(P, R)
        theta5_1, theta5_2 = self.__solve_theta_5(R, theta3)
        theta4 = self.__solve_theta_4(R, theta3, theta5_1)
        theta6 = self.__solve_theta_6(R, theta3, theta5_1)
        return [theta3, theta4, theta5_1, theta6]
