import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGTrainer:
    def __init__(self, policy, args, device=torch.device("cpu")):
        self.policy = policy  # policy.actor, policy.critic, target networks 포함
        self.gamma = args.gamma
        self.tau = 0.05  # soft update coefficient
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm
        self.device = device
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=args.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=args.lr_critic
        )

    def train(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return  # 아직 학습 불가

        obs, actions, rewards, next_obs, dones = replay_buffer.sample(self.batch_size)

        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.policy.target_actor(next_obs)
            target_q = self.policy.target_critic(next_obs, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        # Critic loss
        current_q = self.policy.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor loss
        actor_actions = self.policy.actor(obs)
        actor_loss = -self.policy.critic(obs, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Soft update of target networks
        self.policy.soft_update(self.tau)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value': current_q.mean().item()
        }


