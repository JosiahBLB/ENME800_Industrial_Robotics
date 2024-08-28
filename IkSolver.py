import numpy as np
from MatTrans import MatOp

# define constants
PI_OVER_2 = np.pi / 2

class IkSolver:
    def __init__(self, dh_table) -> None:
        self.a_i_minus_1 = dh_table[0]
        self.alpha_i_minus_1 = dh_table[1]
        self.d_i = dh_table[2]

    def __get_wrist_from_tooltip(self, P:np.ndarray, R:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    def __solve_theta3(self, P:np.ndarray) -> float:
        # rename variables for readability
        a1 = self.a_i_minus_1[1]
        a2 = self.a_i_minus_1[2]
        a3 = self.a_i_minus_1[3]
        d1 = self.d_i[0]
        d4 = self.d_i[3]

        # prep equation 
        x_03 = a1
        z_03 = d1 + a2
        x_06 = P[0]
        z_06 = P[2]

        # use trig identity to solve
        a = z_06 - z_03
        b = x_06 - x_03
        c = d4
        d = a3 
        theta3_1 = np.atan2(a*d - b*c, a*c + b*d)
        
        print(f"theta3_1: {np.rad2deg(theta3_1)}")
        return np.rad2deg(theta3_1)

    def __solve_theta_4(self, R:np.ndarray, theta5:float) -> float:
        # rename variables for readability
        s5 = np.sin(theta5)
        r22 = R[1, 1]
        r21 = R[1, 0]
        theta4 = np.arctan2(r22/-s5, r21/s5)

        print(f"theta4: {np.rad2deg(theta4)}")
        return np.rad2deg(theta4)

    def __solve_theta_5(self, R:np.ndarray, theta3:float) -> float:
        # rename variables for readability
        r13 = R[0, 2]
        r33 = R[2, 2]

        c3 = np.cos(theta3)
        s3 = np.sin(theta3)
        b = r13*c3 - r33*s3
        theta5 = np.arctan2(b, np.sqrt(1 - b**2))

        print(f"theta5: {np.rad2deg(theta5)}")
        return theta5

    def __solve_theta_6(self, R:np.ndarray, theta5:float) -> float:
        # rename variables for readability
        r13 = R[0, 2]
        r33 = R[2, 2]
        s5 = np.sin(theta5)

        # solve for theta_6
        theta6 = np.arctan2(r33/s5, r13/-s5)
        print(f"theta6: {np.rad2deg(theta6)}")
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
        theta3 = self.__solve_theta3(P)
        theta5 = self.__solve_theta_5(R, theta3)
        theta4 = self.__solve_theta_4(R, theta5)
        theta6 = self.__solve_theta_6(R, theta5)
        return [theta3, theta4, theta5, theta6]
