 
from isaaclab.tasks.g1_walk.g1_task import G1WalkEnv
from isaaclab.sim.physx import PhysxCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.config.task_config_base import BaseConfig

G1_CFG = BaseConfig(
    name="G1WalkTask",
    physics=PhysxCfg(),
    scene={
        "robot": ArticulationCfg(
            prim_path="/World/G1",
            usd_path="C:/Users/jlin3/robotix/isaacsimmodels/robots/g1.usd",
            joint_names=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
            ],
            control_mode="position"
        )
    },
    task=G1WalkEnv,
    observations=["joint_pos", "joint_vel"],
    rewards=["base_forward_velocity"],
    termination_conditions=["fall"],
)



# ["pelvis", "pelvis_contour_link",
#                          "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
#                          "left_knee_link", "left_ankle_pitch_link",  "left_ankle_roll_link",
#                          "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
#                          "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
#                          "torso_link", "head_link", "imu_link",
#                          "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
#                          "left_elbow_pitch_link", "left_elbow_roll_link", "logo_link",
#                          "lef_palm_link", "left_six_link", "left_five_link", "left_four_link", 
#                          "left_three_link", "left_two_link", "left_one_link", "left_zero_link",
#                           "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
#                          "right_elbow_pitch_link", "right_elbow_roll_link", "right_palm_link",
#                          "right_six_link", "right_five_link", "right_four_link", "right_three_link",
#                          "right_two_link", "right_one_link", "right_zero_link"],


# all_joint_names = [
#                         "left_hip_pitch_joint","pelvis_contour_joint","right_hip_pitch_joint",
#                         "torso_joint","right_two_joint","left_hip_roll_joint","left_hip_yaw_joint",
#                         "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
#                         "right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
#                         "right_ankle_pitch_joint","right_ankle_roll_joint","head_joint","imu_joint",
#                         "left_shoulder_pitch_joint","logo_joint","right_shoulder_pitch_joint",
#                         "left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_pitch_joint",
#                         "left_elbow_roll_joint","left_palm_joint","left_five_joint",
#                         "left_three_joint","left_zero_joint","left_six_joint","left_four_joint",
#                         "left_one_joint","left_two_joint","right_shoulder_roll_joint",
#                         "right_shoulder_yaw_joint","right_elbow_pitch_joint","right_elbow_roll_joint",
#                         "right_palm_joint","right_five_joint","right_three_joint","right_zero_joint",
#                         "right_six_joint","right_four_joint","right_one_joint","right_two_joint"
#                     ]