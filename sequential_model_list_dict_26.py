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
    def __init__(self, input_dim, output_dim, n_layers=3):
        super().__init__()
        # self.layers = nn.ModuleList(
        #     [nn.Linear(input_dim // n, input_dim // (n+1)) for n in range(1, n_layers+1)]
        # )
        # self.layers = [nn.Linear(input_dim // n, input_dim // (n+1)) for n in range(1, n_layers+1)]
        # self.net = nn.Sequential(*self.layers)
        self.layers = nn.ModuleList()
        for n in range(1, n_layers + 1):
            self.layers.add_module(f'layer_{n}', nn.Linear(input_dim // n, input_dim // (n+1)))
        
        # sz_input = self.layers[-1].out_features
        # self.layer_out = nn.Linear(sz_input, output_dim)
        self.layer_out = nn.Linear(input_dim // (n_layers + 1), output_dim)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.functional.tanh(x)

        x = self.layer_out(x)
        return x
        
model_1 = nn.Sequential(
    nn.Linear(28 * 28, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

model = nn.Sequential()
model.add_module('layer_1', nn.Linear(28 * 28, 32))
model.add_module('relu', nn.ReLU())
model.add_module('layer_2', nn.Linear(32, 10))

model_3 = ModelNN(28*28, 10, 3)
print(model_3.parameters())
print(model_3)