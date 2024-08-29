import numpy as np
from MatOp import MatOp


class FkSolver:
    def __init__(self, dh_table) -> None:
        self.a_i_minus_1 = dh_table[0]
        self.alpha_i_minus_1 = dh_table[1]
        self.d_i = dh_table[2]
        self.N_FRAMES = len(self.d_i)

    def solve_fk(self, *args):
        # ensure correct method usage
        if len(args) == 6:
            theta_i = args
        elif len(args) == 1:
            theta_i = args[0]
        else:
            raise ValueError("Invalid number of arguments.")

        # create dh transformation matrices
        dh_matrices = []
        for i in range(self.N_FRAMES):
            dh_matrices.append(
                MatOp.dh_transform(
                    self.a_i_minus_1[i],
                    self.alpha_i_minus_1[i],
                    self.d_i[i],
                    theta_i[i],
                )
            )
        return dh_matrices
