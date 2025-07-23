from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from isaaclab_assets import GR1T2_CFG


@configclass
class GR1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_roll_link", "right_foot_roll_link"]),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_roll_link", "right_foot_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_roll_link", "right_foot_roll_link"]),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                ]

        )},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[
                "L_index_.*",
                "L_middle_.*",
                "L_pinky_.*",
                "L_ring_.*",
                "L_thumb_.*",
                "R_index_.*",
                "R_middle_.*",
                "R_pinky_.*",
                "R_ring_.*",
                "R_thumb_.*",
            ]

        )},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint"])},
    )


@configclass
class GR1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: GR1Rewards = GR1Rewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = GR1T2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_roll_link"

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)  # reset joints to zero position (neutral)

        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["waist_roll_link"]

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_pitch_joint"])
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_pitch_joint", ".*_ankle_.*"]
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["waist_roll_link"]


@configclass
class GR1RoughEnvCfg_PLAY(GR1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

# 0: left_hip_roll_joint 
# 1: left_hip_yaw_joint
#  2: left_hip_pitch_joint 
# 3: left_knee_pitch_joint
#  4: left_ankle_pitch_joint
#  5: left_ankle_roll_joint
#  6: right_hip_roll_joint 
# 7: right_hip_yaw_joint 
# 8: right_hip_pitch_joint
#  9: right_knee_pitch_joint
# 10: right_ankle_pitch_joint
#  11: right_ankle_roll_joint
#  12: waist_yaw_joint
#  13: waist_pitch_joint
#  14: waist_roll_joint
#  15: head_pitch_joint
#  16: head_roll_joint
#  17: head_yaw_joint
#  18: left_shoulder_pitch_joint
#  19: left_shoulder_roll_joint 
# 20: left_shoulder_yaw_joint 
# 21: left_elbow_pitch_joint 
# 22: left_wrist_yaw_joint 
# 23: left_wrist_roll_joint 
# 24: left_wrist_pitch_joint 
# 25: right_shoulder_pitch_joint 
# 26: right_shoulder_roll_joint 
# 27: right_shoulder_yaw_joint 
# 28: right_elbow_pitch_joint 2
# 9: right_wrist_yaw_joint 
# 30: right_wrist_roll_joint 
# 31: right_wrist_pitch_joint