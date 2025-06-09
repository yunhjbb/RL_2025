import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256, 256]):
        super(DDPGActor, self).__init__()
        layers = []
        input_size = obs_dim
        hidden_sizes = list(map(int, hidden_sizes)) 
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU나 LeakyReLU 사용 많음
            input_size = hidden_size
        self.base = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_size, act_dim)
        self.activation = nn.Tanh()  # action 범위 [-1, 1]

    def forward(self, obs):
        x = self.base(obs)
        action = self.activation(self.output_layer(x))
        return action  # deterministic policy

