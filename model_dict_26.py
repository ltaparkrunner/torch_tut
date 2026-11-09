import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import ImageFolder

class ModelNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=3, act_type=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act_type = act_type
        for n in range(1, n_layers + 1):
            self.layers.add_module(f'layer_{n}', nn.Linear(input_dim // n, input_dim // (n+1)))

        self.layer_out = nn.Linear(input_dim // (n_layers + 1), output_dim)
        self.act_lst = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lk_relu': nn.LeakyReLU()
        })


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.act_type and self.act_type in self.act_lst:
                x = self.act_lst[self.act_type](x)

        x = self.layer_out(x)
        return x

model_3 = ModelNN(28*28, 10, act_type='relu')
print(model_3.parameters())
print(model_3)
