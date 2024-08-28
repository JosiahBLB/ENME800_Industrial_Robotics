# python packages
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import plot_vector

# local packages
from IkSolver import IkSolver
from FkSolver import FkSolver
from MatTrans import MatOp

# define constants
PI_OVER_2 = np.pi / 2


class AbbRobot:
    N_FRAMES = 6

    def __init__(self) -> None:
        tool_length = 0.080  # m
        d_6t = 0.085 + tool_length  # m
        # DH Parameter specific to the 6R ABB robotic arm
        self.a_i_minus_1 = [0, 0.15, 0.9, 0.115, 0, 0, 0]
        self.alpha_i_minus_1 = [0, -PI_OVER_2, 0, -PI_OVER_2, PI_OVER_2, -PI_OVER_2, 0]
        self.d_i = [0.445, 0, 0, 0.795, 0, 0, d_6t]
        dh_table = [self.a_i_minus_1, self.alpha_i_minus_1, self.d_i]
        self.fk_solver = FkSolver(dh_table)
        self.ik_solver = IkSolver(dh_table)

    def move_to_point(self, T):
        if theta_i := self.ik_solver.solve_ik(T):
            # 2 given angles, 4 calculated angles, and the tool orientation
            theta_i = [0, -PI_OVER_2] + theta_i + [0]
            dh_matrices = MatOp.dh_matrices(
                self.a_i_minus_1, self.alpha_i_minus_1, self.d_i, theta_i
            )
            self.plot_robot(dh_matrices)

    def move_to_angles(self, theta_i):
        if dh_matrices := self.fk_solver.solve_fk(theta_i):
            self.plot_robot(dh_matrices)

    def plot_robot(self, dh_matrices):
        rel_dh_matrices = [dh_matrices[0]]

        # plot frames
        pr.plot_basis(R=np.eye(3), label="{0}")
        for i in range(1, AbbRobot.N_FRAMES):
            rel_dh_matrices.append(rel_dh_matrices[i - 1] @ dh_matrices[i])
            pr.plot_basis(
                R=rel_dh_matrices[i][0:3, 0:3],
                p=rel_dh_matrices[i][0:3, 3].T,
                label=f"{i+1}",
            )

        # plot vectors between frames
        colours = ["Red", "Green", "Blue"] * 3
        points = [np.eye(4)] + rel_dh_matrices
        for i in range(len(points)):
            if i < AbbRobot.N_FRAMES:
                direction = points[i + 1][0:3, 3] - points[i][0:3, 3]
                plot_vector(
                    start=points[i][0:3, 3], direction=direction, color=colours[i]
                )
        plt.show()


if __name__ == "__main__":
    abb_robot = AbbRobot()
    T_06 = np.eye(4)
    T_06[0, 3] = 2.652
    T_06[2, 3] = 21.4
    T_06[0:3, 0:3] = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    abb_robot.move_to_point(T_06)
    # abb_robot.move_to_angles([0, 0, 0, 0, 0, 0])
