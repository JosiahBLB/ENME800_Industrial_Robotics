# python packages
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import plot_vector

# local packages
from IkSolver import IkSolver
from FkSolver import FkSolver
from MatOp import MatOp

# define constants
PI_OVER_2 = np.pi / 2


class AbbRobot:
    N_FRAMES = 6
    TEST_SCALE = 3

    def __init__(self) -> None:
        tool_length = 0.080  # m
        d_6t = 0.085 + tool_length  # m
        # DH Parameter specific to the 6R ABB robotic arm
        self.a_i_minus_1 = [0, 0.15, 0.9, 0.115, 0, 0, 0]  # m
        self.alpha_i_minus_1 = [0, -PI_OVER_2, 0, -PI_OVER_2, PI_OVER_2, -PI_OVER_2, 0]
        self.d_i = [0.445, 0, 0, 0.795, 0, 0, d_6t]  # m
        dh_table = [self.a_i_minus_1, self.alpha_i_minus_1, self.d_i]
        for i in range(len(self.a_i_minus_1)):
            self.a_i_minus_1[i] *= AbbRobot.TEST_SCALE
            self.d_i[i] *= AbbRobot.TEST_SCALE
        self.fk_solver = FkSolver(dh_table)
        self.ik_solver = IkSolver(dh_table)

    def __print_final_results(self, theta_i, dh_matrices):
        DH_06 = reduce(lambda a, b: a @ b, dh_matrices).round(3)
        theta_i = [round(float(theta), 3) for theta in theta_i]
        print("End effector position:")
        print(f"px: {DH_06[0, 3]}")
        print(f"py: {DH_06[1, 3]}")
        print(f"pz: {DH_06[2, 3]}")
        print(f"End effector orientation:\n{DH_06[0:3, 0:3]}")
        for i in range(len(theta_i)):
            print(f"J{i+1}: {theta_i[i]}")

    def move_to_point(self, T):
        for i in range(3):
            T[i, 3] *= AbbRobot.TEST_SCALE
        if theta_i := self.ik_solver.solve_ik(T):
            # 2 given angles, 4 calculated angles, and the tool orientation
            theta_i = [0, -PI_OVER_2] + theta_i + [0]
            theta_i = [MatOp.wrap_angle(theta) for theta in theta_i]
            dh_matrices = MatOp.dh_matrices(
                self.a_i_minus_1, self.alpha_i_minus_1, self.d_i, theta_i
            )
            self.__print_final_results(theta_i, dh_matrices)
            self.plot_robot(dh_matrices)

    def move_to_angles(self, theta_i):
        if dh_matrices := self.fk_solver.solve_fk(theta_i):
            theta_i = [MatOp.wrap_angle(theta) for theta in theta_i]
            theta_i = [round(float(theta), 3) for theta in theta_i]
            self.__print_final_results(theta_i, dh_matrices)
            self.plot_robot(dh_matrices)

    def plot_robot(self, dh_matrices):
        rel_dh_matrices = [dh_matrices[0]]

        # plot frames
        pr.plot_basis(R=np.eye(3), label="{0}")
        for i in range(1, AbbRobot.N_FRAMES + 1):
            rel_dh_matrices.append(rel_dh_matrices[i - 1] @ dh_matrices[i])
            pr.plot_basis(
                R=rel_dh_matrices[i][0:3, 0:3],
                p=rel_dh_matrices[i][0:3, 3].T,
                label=f"{i+1 if i < AbbRobot.N_FRAMES else '{T}'}",
            )

        # plot vectors between frames
        colours = ["Red", "Green", "Blue"] * 3
        points = [np.eye(4)] + rel_dh_matrices
        for i in range(len(points)):
            if i < AbbRobot.N_FRAMES + 1:
                direction = points[i + 1][0:3, 3] - points[i][0:3, 3]
                plot_vector(
                    start=points[i][0:3, 3], direction=direction, color=colours[i]
                )
        plt.show()


if __name__ == "__main__":
    abb_robot = AbbRobot()
    T_06 = np.eye(4)

    input = int(input("Enter 1, 2, or 4: "))
    if input == 1:
        # # q1 need to add -pi/2
        T_06[0, 3] = 0.748
        T_06[2, 3] = 2.105
        T_06[0:3, 0:3] = [[-0.707, 0.0, 0.707], [0.0, -1.0, 0.0], [0.707, 0.0, 0.707]]
    elif input == 2:
        # q2 needed to add pi/2
        T_06[0, 3] = -0.61
        T_06[2, 3] = 1.943
        T_06[0:3, 0:3] = [[-0.707, 0.0, -0.707], [0.0, -1.0, 0.0], [-0.707, 0.0, 0.707]]
    else:
        # # q4 need to add pi/2
        T_06[0, 3] = 0.91
        T_06[2, 3] = 0.747
        T_06[0:3, 0:3] = [[0.707, 0.0, 0.707], [0.0, -1.0, 0.0], [0.707, 0.0, -0.707]]

    abb_robot.move_to_point(T_06)

    # abb_robot.move_to_angles(
    #     [0, -PI_OVER_2, PI_OVER_2 / 2, 0, 0, 0, 0]
    # )  # The zero position

    # abb_robot.move_to_angles(
    #     [0, -PI_OVER_2, -PI_OVER_2 / 2, 0, 0, 0, 0]
    # )  # The zero position

    # abb_robot.move_to_angles(
    #     [0, -PI_OVER_2, -PI_OVER_2-PI_OVER_2/2, 0, 0, 0, 0]
    # )  # The zero position

    # abb_robot.move_to_angles([0, -PI_OVER_2, 4.709, -PI_OVER_2, -0.004, 0, 0]) # The zero position
