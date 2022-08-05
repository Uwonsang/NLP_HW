import torch
import torch.nn as nn
import numpy as np
import copy

'''Task2.2 4.Bidirectional Input Flipping'''
def flipping(data_set):

    data_set_copy = copy.deepcopy(data_set)
    flip_tensor_dataset = []

    word_max_len = max(len(data) for data in data_set_copy)

    for data in data_set_copy:
        nonzero_index = torch.nonzero(data)
        flip_np = np.flip(data[nonzero_index].numpy(), 0).copy().reshape(-1)
        flip_np = np.append (flip_np, [0] * (word_max_len - len(flip_np)))
        flip_tensor_dataset.append(flip_np)

    flip_tensor_final = torch.LongTensor(np.array(flip_tensor_dataset))

    return flip_tensor_final


def masking(data_set):
    masked = data_set.gt(0)
    return masked

class Embedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        x = self.weight[x]
        return x


class simple_RNN(nn.Module):
    def __init__(self, embed_weight_vec, device):
        super(simple_RNN, self).__init__()
        self.input_dim = 300
        self.hidden_dim = 300
        self.output_dim = 6
        self.device = device
        self.embedding = Embedding(embed_weight_vec)

        self.encoder = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        self.fc1 = nn.Sequential(
                    nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )


    def forward(self, x):
        batch_size = x.size()[0]
        fc1_hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc2_hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc3_hidden = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        x = self.embedding(x) # batch_size(256), time_size(20), input_size(300)
        x_p = x.permute(1, 0, 2) # time_size(20), batch_size(256), input_size(300)

        output = []
        for i in range(x_p.size()[0]):
            fc1_input = torch.cat([x_p[i],  fc1_hidden], dim=1)
            fc1_hidden = self.fc1(fc1_input)
            fc2_input = torch.cat([fc1_hidden, fc2_hidden], dim=1)
            fc2_hidden = self.fc2(fc2_input)
            fc3_input = torch.cat([fc2_hidden, fc3_hidden], dim=1)
            fc3_hidden = self.fc3(fc3_input)
            output.append(fc3_hidden)
        output = torch.stack(output, dim=0)
        outs = self.decoder(output[-1])

        return outs


class Bidirectional_RNN(nn.Module):
    def __init__(self, embed_weight_vec, device):
        super(Bidirectional_RNN, self).__init__()
        self.input_dim = 300
        self.hidden_dim = 300
        self.output_dim = 18
        self.device = device
        self.embedding = Embedding(embed_weight_vec)

        self.decoder = nn.Linear(self.hidden_dim + self.hidden_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

        self.fc1_L2R = nn.Sequential(
                    nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.2)
        )

        self.fc2_L2R = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)

        )

        self.fc3_L2R = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.fc1_R2L = nn.Sequential(
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.fc2_R2L = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )

        self.fc3_R2L = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2)
        )


    def forward(self, x):
        batch_size = x.size()[0]
        fc1_hidden_L2R = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc2_hidden_L2R = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc3_hidden_L2R = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        fc1_hidden_R2L = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc2_hidden_R2L = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        fc3_hidden_R2L = torch.zeros(batch_size, self.hidden_dim).to(self.device)

        # x = masking(x)
        x_L2R = x.to(self.device)
        x_L2R = self.embedding(x_L2R) # batch_size(256), time_size(20), input_size(300)
        x_L2R_p = x_L2R.permute(1, 0, 2) # time_size(20), batch_size(256), input_size(300)

        x_R2L = flipping(x)
        # x_R2L = masking(x_R2L)
        x_R2L = x_R2L.to(self.device)
        x_R2L = self.embedding(x_R2L)  # batch_size(256), time_size(20), input_size(300)
        x_R2L_p = x_R2L.permute(1, 0, 2)  # time_size(20), batch_size(256), input_size(300)

        time_size = x_L2R_p.size()[0]
        output = []
        for i in range(time_size):
            ## Left to Right
            fc1_input_L2R = torch.cat([x_L2R_p[i],  fc1_hidden_L2R], dim=1)
            fc1_hidden_L2R = self.fc1_L2R(fc1_input_L2R)
            fc2_input_L2R = torch.cat([fc1_hidden_L2R, fc2_hidden_L2R], dim=1)
            fc2_hidden_L2R = self.fc2_L2R(fc2_input_L2R)
            fc3_input_L2R = torch.cat([fc2_hidden_L2R, fc3_hidden_L2R], dim=1)
            fc3_hidden_L2R = self.fc3_L2R(fc3_input_L2R)

            ## Right to Left
            fc1_input_R2L = torch.cat([x_R2L_p[i],  fc1_hidden_R2L], dim=1)
            fc1_hidden_R2L = self.fc1_R2L(fc1_input_R2L)
            fc2_input_R2L = torch.cat([fc1_hidden_R2L, fc2_hidden_R2L], dim=1)
            fc2_hidden_R2L = self.fc2_R2L(fc2_input_R2L)
            fc3_input_R2L = torch.cat([fc2_hidden_R2L, fc3_hidden_R2L], dim=1)
            fc3_hidden_R2L = self.fc3_R2L(fc3_input_R2L)
            ## concat
            hidden_L2R_R2L = torch.cat((fc3_hidden_L2R, fc3_hidden_R2L), 1)
            output.append(hidden_L2R_R2L)
        output = torch.stack(output, dim=0)
        outs = self.decoder(output)  # time_size(20), batch_size(256), output_size(18)
        outs = outs.permute(1, 0, 2) # batch_size(256), time_size(20), output_size(18)

        return outs