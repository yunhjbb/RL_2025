import torch
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic

class PPOPolicy:
    def __init__(self, obs_dim, act_dim, args, device=torch.device("cpu")):
        #self, obs_dim, act_dim, args, hidden_size_actor=64, hidden_size_critic = 64, device=torch.device("cpu")
        self.device = device
        arch_actor = args.arch_actor
        arch_critic = args.arch_critic
        if arch_actor == 'MLP':
            self.actor = PPOActor(obs_dim, act_dim, args.hidden_size_actor).to(device)
        else:
            raise NotImplementedError
        if arch_critic == 'MLP':
            self.critic = PPOCritic(obs_dim, args.hidden_size_critic).to(device)
        else:
            raise NotImplementedError

    def get_actions(self, obs, deterministic=False):
        obs = obs.to(self.device)
        actions, log_probs = self.actor(obs, deterministic)
        values = self.critic(obs)
        return values, actions, log_probs

    def evaluate_actions(self, obs, actions):
        obs, actions = obs.to(self.device), actions.to(self.device)
        log_probs, entropy = self.actor.evaluate_actions(obs, actions)
        values = self.critic(obs)
        return values, log_probs, entropy

    def get_values(self, obs):
        obs = obs.to(self.device)
        return self.critic(obs)

    def act(self, obs, deterministic=False):
        obs = obs.to(self.device)
        actions, _ = self.actor(obs, deterministic)
        return actions

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def copy(self):
        new_policy = PPOPolicy(
            obs_dim=self.actor.base[0].in_features,
            act_dim=self.actor.mu_head.out_features,
            hidden_size=self.actor.base[0].out_features,
            device=self.device
        )
        new_policy.actor.load_state_dict(self.actor.state_dict())
        new_policy.critic.load_state_dict(self.critic.state_dict())
        return new_policy
