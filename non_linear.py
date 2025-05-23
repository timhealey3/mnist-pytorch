import torch.nn as nn
import torch
from torch import matmul

class NonLinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        print("NonLinearNN")
        # defining weights and biases
        # 128
        self.weight1 = nn.Parameter(torch.rand(784, 128))
        self.bias1 = nn.Parameter(torch.rand(128))
        # 64
        self.weight2 = nn.Parameter(torch.rand(128, 64))
        self.bias2 = nn.Parameter(torch.rand(64))
        # 32
        self.weight3 = nn.Parameter(torch.rand(64, 32))
        self.bias3 = nn.Parameter(torch.rand(32))
        # 16
        self.weight4 = nn.Parameter(torch.rand(32, 16))
        self.bias4 = nn.Parameter(torch.rand(16))
        # 10
        self.weight5 = nn.Parameter(torch.rand(16, 10))
        self.bias5 = nn.Parameter(torch.rand(10))
        # non linearity activation function
        self.relu = nn.ReLU()
        # flatten
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        # flatten input
        x = self.flatten(x)
        # layer 1
        x_1 = matmul(x, self.weight1) + self.bias1
        z_1 = self.relu(x_1)
        # layer 2
        x_2 = matmul(z_1, self.weight2) + self.bias2
        z_2 = self.relu(x_2)
        # layer 3
        x_3 = matmul(z_2, self.weight3) + self.bias3
        z_3 = self.relu(x_3)
        # layer 4
        x_4 = matmul(z_3, self.weight4) + self.bias4
        z_4 = self.relu(x_4)
        # output
        x_5 = matmul(z_4, self.weight5) + self.bias5
        
        return x_5