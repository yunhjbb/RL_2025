import torch
import torch.nn as nn

class PPOTrainer:
    def __init__(self, policy, args, device=torch.device("cpu")):
        self.policy = policy
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.batch_size
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.device = device
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        '''
        self.optimizer = torch.optim.Adam(
            list(self.policy.actor.parameters()) + list(self.policy.critic.parameters()), lr=lr
        )
        '''
        # optimizer 분리
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr_critic)

    def train(self, obs, actions, old_log_probs, returns, advantages):
        self.policy.train()
        for _ in range(self.ppo_epoch):
            indices = torch.randperm(obs.size(0))
            for start in range(0, obs.size(0), self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                

                values, log_probs, entropy = self.policy.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()
                if _ == 0 and start == 0:
                    print(f"policy loss : {policy_loss}")
                    print(f"value loss: {value_loss}")
                    print(f"entropy loss: {self.entropy_coef * entropy_loss}")
                '''
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                '''
                # 각각 역전파
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # actor → policy + entropy 부분 포함
                actor_total_loss = policy_loss + self.entropy_coef * entropy_loss
                actor_total_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # critic → value loss만
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

