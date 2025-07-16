from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom
import numpy as np
from scipy.spatial.transform import Rotation as R

def perform_vr_raycast(start_pos: tuple, direction: tuple, max_distance: float = 10.0):
    """
    Casts a ray from start_pos along direction, returns hit position in world coordinates or None.
    
    Args:
        start_pos (tuple): (x, y, z) start point of ray in world coords.
        direction (tuple): (x, y, z) normalized direction vector of ray.
        max_distance (float): max ray length.

    Returns:
        hit_pos_world (tuple) or None
    """
    stage = get_current_stage()
    if not stage:
        print("Stage not loaded")
        return None

    from omni.isaac.core import SimulationContext
    sim = SimulationContext.instance()

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
    """
    Converts world hit position to robot local frame.

    Args:
        hit_pos_world (tuple): (x,y,z) hit point in world coords
        robot_pos_world (tuple): (x,y,z) robot base position in world coords
        robot_orientation_quat (tuple): (x,y,z,w) robot orientation quaternion in world frame

    Returns:
        np.array: hit position in robot's local frame
    """
    hit_vec = np.array(hit_pos_world)
    robot_pos = np.array(robot_pos_world)
    rot = R.from_quat(robot_orientation_quat)
    # Inverse rotation to go from world to local
    local_pos = rot.inv().apply(hit_vec - robot_pos)
    return local_pos

def compute_velocity_command(local_target, max_speed=1.0):
    # Simplistic proportional controller to walk towards the target
    direction = local_target.copy()
    direction[2] = 0  # ignore height for walking
    dist = np.linalg.norm(direction)
    if dist < 0.1:
        return np.array([0.0, 0.0])  # stop if close enough

    direction_norm = direction / dist
    velocity_command = direction_norm * min(dist, max_speed)
    # Returns (vx, vy) desired base linear velocities in robot frame
    return velocity_command[:2]
