"""Robot configuration dataclass that works with and without Hydra."""

from omegaconf import DictConfig, OmegaConf
import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class RobotCfg:
    """Robot parameters required to generate symmetry group representations."""

    name: str = "???"
    floating_base: bool = True  # Whether the robot's base has free floating joint(nq=7, nv=6)

    # Symmetry related parameters
    group_label: Optional[str] = None

    joint_space_order: Optional[List[str]] = None  # Names of the joints (ignoring floating base)
    kinematic_chains: Optional[List[List[str]]] = None  # Names of kinematic chains.

    permutation_Q_js: Optional[List[List[int]]] = None  # Symmetry permutation of joint-space position coordinates
    permutation_TqQ_js: Optional[List[List[int]]] = None  # Symmetry permutations of joint-space tangent space
    reflection_Q_js: Optional[List[List[int]]] = None  # Symmetry reflections of joint-space position coordinates
    reflection_TqQ_js: Optional[List[List[int]]] = None  # Symmetry reflections of joint-space tangent space
    permutation_kin_chain: Optional[List[List[int]]] = None  # Symmetry permutations of kinematic chains
    reflection_kin_chain: Optional[List[List[int]]] = None  # Symmetry reflections of kinematic chains

    fix_base: bool = False

    # Visualization auxiliary variables
    hip_height: float = 1.0
    endeff_names: Optional[List[str]] = None
    q_zero: Optional[List[float]] = None  # Zero vector of generalized positions

    # URDF params
    urdf_path: Optional[str] = None
    # MJCF params
    mjcf_path: Optional[str] = None
    variant: Optional[str] = None
    commit: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to handle default values and cross-references."""
        # Handle cross-references from YAML (${.permutation_Q_js} style)
        if self.permutation_TqQ_js is None and self.permutation_Q_js is not None:
            self.permutation_TqQ_js = self.permutation_Q_js

        if self.reflection_TqQ_js is None and self.reflection_Q_js is not None:
            self.reflection_TqQ_js = self.reflection_Q_js

    def as_dict_cfg(self) -> DictConfig:
        """Convert the dataclass to a DictConfig."""
        dict = dataclasses.asdict(self)
        return OmegaConf.create(dict)
