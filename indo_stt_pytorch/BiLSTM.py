import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)

        return x
