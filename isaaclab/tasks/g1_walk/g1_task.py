import torch
from isaaclab.envs.base_env import RLTaskEnv

class G1WalkEnv(RLTaskEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.robot = self.scene.robots[0]  # one g1 robot in the scene

    def pre_physics_step(self, actions: torch.Tensor):
        self.robot.set_joint_position_targets(actions)

    def get_observations(self) -> dict: # finds the joint positions and velocities
        return {
            "joint_pos": self.robot.data.joint_pos.clone(),
            "joint_vel": self.robot.data.joint_vel.clone(),
        }

    def get_reward(self) -> torch.Tensor: # a positive x means forward movement = reward
        base_lin_vel = self.robot.data.root_lin_vel_w 
        forward_vel = base_lin_vel[:, 0]
        return forward_vel

    def is_done(self) -> torch.Tensor: # if the base height is less than 0.15, the robot has technically fallen
        base_height = self.robot.data.root_pos_w[:, 2]
        done = base_height < 0.15
        return done
