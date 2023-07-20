import torch
import torch.nn as nn
import torch.optim as optim

# ---------- THIS MODEL IS ON PROGRESS ----------------- #

class RNN_T(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN_T, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, inputs):
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)
        outputs = self.softmax(outputs)
        return outputs
