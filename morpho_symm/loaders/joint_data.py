from dataclasses import dataclass, field


@dataclass
class JointData:
    """Dataclass to store information about the joints of a robot.

    Attributes:
    ----------
    name : (str) The name of the joint.
    type : (int) The type of the joint.
    body_id : (int) The body id of the joint.
    nq : (int) The number of generalized coordinates.
    nv : (int) The number of generalized velocities.
    qpos_idx : (tuple) The indices of the joint's generalized coordinates.
    qvel_idx : (tuple) The indices of the joint's generalized velocities.
    tau_idx: (tuple) The indices of the joint's in the generalized forces vector.
    actuator_id: (int) The id of the actuator associated with the joint.
    range : list(min, max) The range of the joint's generalized coordinates.
    """

    name: str
    type: int
    body_id: int
    nq: int
    nv: int
    qpos_idx: tuple
    qvel_idx: tuple
    range: list
    tau_idx: tuple = field(default_factory=tuple)
    actuator_id: int = field(default=-1)

    def __str__(self):
        """Return a string representation of the joint information."""
        return f"{', '.join([f'{key}={getattr(self, key)}' for key in self.__dict__])}"
