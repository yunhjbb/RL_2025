import argparse

# 추가로 필요한 것들
#hidden size 전달달
def get_config():
    """
    Parses config including XML path, max steps, CUDA usage,
    and standard PPO training arguments.
    """
    parser = argparse.ArgumentParser(description="Minimal config for Mujoco PPO environment")

    # Render arguments
    parser.add_argument('--act-by-model', action='store_true', default=False,
                        help="Render robot by actor or not")

    # Mujoco-specific arguments
    parser.add_argument('--xml-path', type=str, required=True,
                        help="Path to Mujoco XML model file")
    parser.add_argument('--max-steps', type=int, default=1000,
                        help="Maximum number of steps per episode")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="Use CUDA if available")

    # Experiment setup
    parser.add_argument('--env-name', type=str, default='SimpleMuJoCoEnv',
                        help="Name of the environment")
    parser.add_argument('--algorithm-name', type=str, default='ppo',
                        help="Name of the RL algorithm to use (e.g., ppo, mappo)")
    parser.add_argument('--experiment-name', type=str, default='default_exp',
                        help="Name to distinguish experiment runs")
    parser.add_argument('--seed', type=int, default=1,
                        help="Random seed for reproducibility")
    parser.add_argument('--num-episodes', type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument('--eval-interval', type=int, default=50,
                        help="Run evaluation every N episodes")
    parser.add_argument('--update-every', type=int, default=10,
                        help="Number of episodes per policy update")
    parser.add_argument('--save-every', type=int, default=20,
                    help="Number of episodes per model saving")
    parser.add_argument('--save-dir', type=str, default='save',
                        help="save model directory")
    parser.add_argument('--env-config-path', type=str, default='..\env_config.yaml',
                        help="environment setting")
    parser.add_argument('--load-actor-path', type=str, default='save/actor_latest.pt',
                        help="model path when continual learning")
    parser.add_argument('--load-critic-path', type=str, default='save/critic_latest.pt',
                        help="model path when continual learning")
    parser.add_argument('--render-model-path', type=str, default='save/actor_latest.pt',
                        help="model path when rendering")
    parser.add_argument('--continual-learning', action='store_true', default=False,
                        help="continue learning by existing model")
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for rewards")
    parser.add_argument('--lam', type=float, default=0.95,
                        help="Discount factor for gae(Generalized advantage estimate)")
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument('--lr_actor', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help="PPO clipping parameter")
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help="Number of PPO update epochs per training step")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Minibatch size for PPO training")
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help="Weight for value loss in PPO")
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help="Weight for entropy loss in PPO")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="Max norm for gradient clipping")

    # NN hyperparameters
    parser.add_argument('--hidden-size-actor', type=int, nargs='+', default=[64, 64],
                        help="hidden feature size for actor MLP")
    parser.add_argument('--hidden-size-critic', type=int, nargs='+', default=[64, 64],
                        help="hidden feature size for critic MLP")   
    parser.add_argument('--arch-actor', type=str, default='MLP',
                        help='nn architecture of actor') 
    parser.add_argument('--arch-critic', type=str, default='MLP',
                        help='nn architecture of critic')     
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help="deterministic action value or not")         
    return parser

if __name__ == "__main__":
    parser = get_config()
    args = parser.parse_args()
    print(args)
