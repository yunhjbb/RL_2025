import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[256, 256]):
        super(PPOCritic, self).__init__()
        layers = []
        input_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())  # 활성화 함수
            input_size = hidden_size  # 다음 입력 크기 갱신
        self.base = nn.Sequential(*layers)
        self.value_out = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        x = self.base(obs)
        value = self.value_out(x)
        return value
