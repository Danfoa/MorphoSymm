# Created by Daniel Ordoñez (daniels.ordonez@gmail.com) at 07/03/25
import pathlib

import pytest

from morpho_symm.utils.robot_utils import load_symmetric_system


@pytest.mark.parametrize("load_pin_robot", [False, True])
def test_robot_import_no_pin(load_pin_robot):  # noqa: D103
    cfg_path = pathlib.Path("morpho_symm/cfg/robot")
    assert cfg_path.exists(), f"Path {cfg_path.absolute()} does not exist"
    all_robot_cfgs = pathlib.Path("morpho_symm/cfg/robot").rglob("*.yaml")

    robot_names = [p.stem for p in all_robot_cfgs]
    robot_names.remove("base_robot")
    for robot_name in robot_names:
        try:
            if load_pin_robot and robot_name == "iiwa":
                continue  # Git clone error unrelated to us.
            load_symmetric_system(robot_name=robot_name, return_robot=load_pin_robot)
        except Exception as e:
            raise AssertionError(f"Failed to load robot {robot_name}") from e


#
# if __name__ == "__main__":
#     test_robot_import_no_pin(load_pin_robot=True)
#     print("All tests passed!")
