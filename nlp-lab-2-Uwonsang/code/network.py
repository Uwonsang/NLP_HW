import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.embedding_size))

        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = self.weight[x]
        return x

class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        x_f = []
        for i in range(len(x)):
            x_f.append(torch.sum(x[i], dim=1))
        x_f = torch.stack(x_f, dim=0)
        return x_f


class simple_CNN(nn.Module):
    def __init__(self, vocab_size, sentence_length, word_length):
        super(simple_CNN, self).__init__()
        self.embedding_dim = 100
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.word_length = word_length

        self.embedding = Embedding(self.vocab_size, self.embedding_dim)
        self.flatten = flatten()

        self.layers_1 = nn.Sequential(
            nn.Conv1d(in_channels= 1000, out_channels=100, kernel_size=2, stride=1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(19) #19
        )

        self.layers_2 = nn.Sequential(
            nn.Conv1d(in_channels= 1000, out_channels=100, kernel_size=3, stride=1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(18) #18
        )

        self.layers_3 = nn.Sequential(
            nn.Conv1d(in_channels= 1000, out_channels=100, kernel_size=4, stride=1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(17) #17
        )

        self.fc1 = nn.Linear(300, 100)
        self.fc2 = nn.Linear(100, 6)


    def forward(self, x):
        x = self.embedding(x) ## batch ,sentence_length, word_length, embedding_dim
        x_p = x.reshape(x.shape[0], 20, -1)  ## batch ,sentence_length, embedding_dim
        x_p = x_p.permute(0, 2, 1)  ## batch ,embedding_dim, sentence_length
        x_1 = self.layers_1(x_p)
        x_2 = self.layers_2(x_p)
        x_3 = self.layers_3(x_p)
        x_total = torch.cat([x_1, x_2, x_3], dim=1).squeeze(dim=2)
        x_total = self.fc1(x_total)
        x_total = torch.relu(x_total)
        x_total = self.fc2(x_total)
        x_total = torch.softmax(x_total, dim=1)
        return x_total