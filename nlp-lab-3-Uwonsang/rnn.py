import torch
import torch.nn as nn

class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1) # (10, 50), #(10, 20)
        hidden = self.i2h(input) # (10, 70) - > (10, 20)
        output = self.h2o(hidden) # (10, 20) -> (10, 10)
        return hidden, output


rnn = RNN(50, 20, 10)


loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = torch.randn(batch_size, 50)
hidden = torch.zeros(batch_size, 20)
target = torch.zeros(batch_size, 10)

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()