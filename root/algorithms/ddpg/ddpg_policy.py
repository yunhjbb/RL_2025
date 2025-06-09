import torch
from .ddpg_actor import DDPGActor
from .ddpg_critic import DDPGCritic

class DDPGPolicy:
    def __init__(self, obs_dim, act_dim, actor_cls, critic_cls, args, device):
        self.device = device
        arch_actor = args.arch_actor
        arch_critic = args.arch_critic
        if arch_actor == 'MLP':
            self.actor = actor_cls(obs_dim, act_dim, args.hidden_size_actor).to(device)
            self.target_actor = actor_cls(obs_dim, act_dim, args.hidden_size_actor).to(device)
        else:
            raise NotImplementedError
        if arch_critic == 'MLP':
            self.critic = critic_cls(obs_dim, act_dim, args.hidden_size_critic).to(device)
            self.target_critic = critic_cls(obs_dim, act_dim, args.hidden_size_critic).to(device)
        else:
            raise NotImplementedError

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def get_action(self, obs, noise_std=0.1, clip_range=1.0):
        obs = obs.to(self.device)
        with torch.no_grad():
            action = self.actor(obs)
        if noise_std > 0:
            noise = noise_std * torch.randn_like(action)
            action = action + noise
        return torch.clamp(action, -clip_range, clip_range)

    def soft_update(self, tau=0.005):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def eval(self):
        self.actor.eval()
        self.critic.eval()

