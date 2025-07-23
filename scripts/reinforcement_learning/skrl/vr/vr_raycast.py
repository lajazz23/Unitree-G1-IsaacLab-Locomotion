
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sim.simulation_context import SimulationContext


from pxr import Gf, UsdGeom
import numpy as np
from scipy.spatial.transform import Rotation as R

def perform_vr_raycast(start_pos: tuple, direction: tuple, max_distance: float = 10.0):

    stage = get_current_stage()
    if not stage:
        print("Stage not loaded")
        return None

    sim = SimulationContext.instance()
    if sim is None:
        sim = SimulationContext()


    scene = sim.get_physics_scene()
    start_point = Gf.Vec3d(*start_pos)
    direction_vec = Gf.Vec3d(*direction).GetNormalized()
    end_point = start_point + direction_vec * max_distance

    hit, hit_point, _, _, _ = scene.raycast(start_point, end_point)

    if hit:
        return (hit_point[0], hit_point[1], hit_point[2])
    else:
        return None


def world_to_robot_local(hit_pos_world, robot_pos_world, robot_orientation_quat):

    hit_vec = np.array(hit_pos_world)
    robot_pos = np.array(robot_pos_world)
    rot = R.from_quat(robot_orientation_quat)
    local_pos = rot.inv().apply(hit_vec - robot_pos)
    return local_pos

def compute_velocity_command(local_target, max_speed=1.0):
    direction = local_target.copy()
    direction[2] = 0  
    dist = np.linalg.norm(direction)
    if dist < 0.1:
        return np.array([0.0, 0.0]) 

    direction_norm = direction / dist
    velocity_command = direction_norm * min(dist, max_speed)
    return velocity_command[:2]
