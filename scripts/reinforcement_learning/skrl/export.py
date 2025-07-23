import os
import torch
from isaaclab_rl.rsl_rl import export_policy_as_onnx
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import gymnasium as gym
from skrl.utils.runner.torch import Runner

def export_policy(checkpoint_path, task_name, algorithm="ppo"):
    algorithm = algorithm.lower()
    
    # Load experiment config
    experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    
    # Prepare env config
    env_cfg = parse_env_cfg(task_name)
    
    # Create environment
    env = gym.make(task_name, cfg=env_cfg)
    
    # Wrap environment for SKRL
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # adjust if using JAX
    
    # Create runner (this creates agent inside)
    runner = Runner(env, experiment_cfg)
    
    # Load the checkpoint into agent
    runner.agent.load(checkpoint_path)
    
    # Export policy to ONNX file
    export_dir = os.path.dirname(checkpoint_path)
    export_policy_as_onnx(policy=runner.agent, path=export_dir, filename="policy.onnx", verbose=True)
    
    print(f"Exported policy to {os.path.join(export_dir, 'policy.onnx')}")

if __name__ == "__main__":
    # Example usage:
    checkpoint = "/path/to/your/checkpoint.pt"
    task = "YourTaskName"
    export_policy(checkpoint, task)
