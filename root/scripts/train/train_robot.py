import sys
import os
target_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(target_path)
from runner.mujoco_runner import SimpleRunner  
from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer
from envs.MujocoSim.envs.robot_env import SimpleMuJoCoEnv
from config import get_config
import traceback
import torch
import mujoco
import yaml

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)
    
def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print(all_args.cuda)
    print(torch.cuda.is_available())

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda")  
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    # env init
    xml_path = all_args.xml_path
    config = load_config(all_args.env_config_path)
    print(config)
    env = SimpleMuJoCoEnv(xml_path, config)
    runner = SimpleRunner(env, all_args, device)

    try:
        runner.run()
    except BaseException:
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])