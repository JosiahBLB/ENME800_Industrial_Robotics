# python packages
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import plot_vector, make_3d_axis
from pytransform3d.transformations import plot_transform
from matplotlib.animation import FuncAnimation

# local packages
from IkSolver import IkSolver
from FkSolver import FkSolver
from MatOp import MatOp
import time

# define constants
PI_OVER_2 = np.pi / 2
PI_OVER_4 = np.pi / 4
PI_3_OVER_4 = np.pi * 3 / 4


class AbbRobot:
    N_FRAMES = 6
    TEST_SCALE = 3
    AX_SCALE = 3

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
        DH_06 = reduce(lambda a, b: a @ b, dh_matrices)
        theta_i = [round(float(theta), 3) for theta in theta_i]
        print("End effector position:")
        print(DH_06)
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
                ax_s=AbbRobot.AX_SCALE,
            )

        # plot vectors between frames
        colours = ["Red", "Green", "Blue"] * 3
        points = [np.eye(4)] + rel_dh_matrices
        for i in range(len(points)):
            if i < AbbRobot.N_FRAMES + 1:
                direction = points[i + 1][0:3, 3] - points[i][0:3, 3]
                plot_vector(
                    start=points[i][0:3, 3],
                    direction=direction,
                    color=colours[i],
                    ax_s=AbbRobot.AX_SCALE,
                )

    @staticmethod
    def theta3(t):
        return np.deg2rad(6.8 - 2.2667 * t**2 + 0.5037 * t**3)

    @staticmethod
    def theta5(t):
        return np.deg2rad(83.7 - 15.6 * t**2 + 3.467 * t**3)

    def plot_a_to_b(self):
        start_time = time.time()
        future_time = start_time + 3

        def update(frame):
            current_time = time.time()
            if current_time >= future_time:
                anim.event_source.stop()
                return

            plt.cla()
            t = current_time - start_time
            print(t)
            theta_1 = 0
            theta_2 = -np.deg2rad(90) + np.deg2rad(52.4)
            theta_3 = AbbRobot.theta3(t)
            theta_4 = 0
            theta_5 = AbbRobot.theta5(t)
            theta_6 = 0
            self.move_to_angles(
                [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, 0]
            )
            plt.draw()

        fig = plt.figure(figsize=(10, 8))
        anim = FuncAnimation(fig, update, interval=100)
        plt.show()


if __name__ == "__main__":
    abb_robot = AbbRobot()
    T_06 = np.eye(4)

    solution_type = input(
        "Enter 1 for IK or 2 for FK solution, otherwise enter to plot A to B:\n"
    )
    if solution_type.isdigit():
        solution_type = int(solution_type)
        position = int(
            input(
                "Enter:\n\t0) Zero\n\t1) Quadrant 1\n\t2) Quadrant 2\n\t4) Quadrant 4\n"
            )
        )
    else:
        abb_robot.plot_a_to_b()
        exit(0)

    if position == 0:
        # the zero position
        T_06 = np.array(
            [
                [6.12323400e-17, 6.12323400e-17, 1.00000000e00, 3.33000000e00],
                [-6.12323400e-17, -1.00000000e00, 6.12323400e-17, -1.01033361e-17],
                [1.00000000e00, -6.12323400e-17, -6.12323400e-17, 4.38000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        theta = [0, -PI_OVER_2, 0, 0, 0, 0, 0]
    elif position == 1:
        # quadrant 1 f
        T_06 = np.array(
            [
                [-7.07106781e-01, 4.32978028e-17, 7.07106781e-01, 2.24251569e00],
                [-4.32978028e-17, -1.00000000e00, 1.79345371e-17, -1.28613593e-16],
                [7.07106781e-01, -1.79345371e-17, 7.07106781e-01, 6.31541937e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        theta = [0, -PI_OVER_2, -PI_OVER_4, 0, 0, 0, 0]
    elif position == 2:
        # quadrant 2
        T_06 = np.array(
            [
                [-7.07106781e-01, -4.32978028e-17, -7.07106781e-01, -1.83041937e00],
                [4.32978028e-17, -1.00000000e00, 1.79345371e-17, -9.87381089e-17],
                [-7.07106781e-01, -1.79345371e-17, 7.07106781e-01, 5.82751569e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        theta = [0, -PI_OVER_2, -PI_3_OVER_4, 0, 0, 0, 0]
    elif position == 4:
        # quadrant 4
        T_06 = np.array(
            [
                [7.07106781e-01, 4.32978028e-17, 7.07106781e-01, 2.73041937e00],
                [-4.32978028e-17, -1.00000000e00, 1.04530143e-16, 1.20781751e-16],
                [7.07106781e-01, -1.04530143e-16, -7.07106781e-01, 2.24248431e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        theta = [0, -PI_OVER_2, PI_OVER_4, 0, 0, 0, 0]

    if solution_type == 1:
        abb_robot.move_to_point(T_06)
    elif solution_type == 2:
        abb_robot.move_to_angles(theta)

    plt.show()
