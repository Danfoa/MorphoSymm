from sympy.interactive.printing import init_printing

from robot_kinematic_symmetries import JointSpaceSymmetry, SemiDirectProduct, is_equivariant_invariant

init_printing(use_unicode=False, wrap_line=False)

from emlp.reps import Vector

from utils.emlp_visualization import *
from utils.utils import *
import torch
torch.set_default_dtype(torch.float32)

if __name__ == "__main__":
    np.random.seed(1)

    permutations_in = [(3, 2, 1, 0),]
    permutations_out = [(2, 1, 0),]

