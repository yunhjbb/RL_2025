import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        super(DDPGCritic, self).__init__()
        layers = []
        input_size = obs_dim + act_dim  # ✅ s, a 모두 입력
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU나 LeakyReLU 자주 사용
            input_size = hidden_size
        self.base = nn.Sequential(*layers)
        self.q_out = nn.Linear(input_size, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)  # ✅ (s, a) 결합
        x = self.base(x)
        q_value = self.q_out(x)
        return q_value  # Q(s, a)

