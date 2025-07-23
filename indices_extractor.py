from omni.isaac.core.utils.stage import get_stage
from omni.isaac.core.articulations import Articulation

stage = get_stage()
robot_path = "/World/envs/env_0/Robot"  # Adjust your robot's path here
robot = Articulation(prim_path=robot_path)

robot.initialize()

for idx, name in enumerate(robot.get_dof_names()):
    print(f"{idx}: {name}")
