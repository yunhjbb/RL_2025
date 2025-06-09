import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes = [256, 256]):
        super(PPOActor, self).__init__()
        layers = []
        input_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())  # 활성화 함수
            input_size = hidden_size  # 다음 입력 크기 갱신
        self.base = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_size, act_dim)
        self.activation = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs, deterministic=False):
        x = self.base(obs)
        mu = self.activation(self.mu_head(x))   
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()
            action = torch.clamp(action, -1.0, 1.0)  # 샘플된 값이 범위 넘는 경우 대비
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        x = self.base(obs)
        mu = self.activation(self.mu_head(x))
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
