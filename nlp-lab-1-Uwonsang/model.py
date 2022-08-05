import torch
import torch.nn as nn
import math
import numpy as np


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.weights = nn.Parameter(torch.randn(self.input_size, self.output_size))
        self.bias = nn.Parameter(torch.randn(self.output_size))

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights.t())
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.matmul(x , self.weights)
        return torch.add(x ,self.bias)



class simple_MLP(nn.Module):
    def __init__(self, input_dim):
        super(simple_MLP, self).__init__()

        self.input_dim = input_dim
        self.fc1 = Linear(self.input_dim, 1000)
        self.fc2 = Linear(1000, 100)
        self.fc3 = Linear(100, 19)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weights = nn.Parameter(torch.randn(self.vocab_size, self.embedding_size))

    def forward(self, x):
        x = self.weights[x]
        return x



class simple_MLP_embed(nn.Module):
    def __init__(self, vocab_size):
        super(simple_MLP_embed, self).__init__()
        self.embedding_dim = 1000
        self.vocab_size = vocab_size

        self.embedding = Embedding(self.vocab_size, self.embedding_dim)
        self.fc1 = Linear(20000, 1000)
        self.fc2 = Linear(1000, 100)
        self.fc3 = Linear(100, 19)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(len(x), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x