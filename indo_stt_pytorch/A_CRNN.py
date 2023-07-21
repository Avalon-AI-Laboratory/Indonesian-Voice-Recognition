import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.att_weights.data.normal_(mean=0.0, std=0.05)
        self.query_vector = nn.Parameter(torch.Tensor(hidden_size))
        self.query_vector.data.normal_(mean=0.0, std=0.05)

    def forward(self, lstm_output):
        if lstm_output.dim() != 3:
            raise RuntimeError("Input tensor must be 3D, got {}D".format(lstm_output.dim()))

        batch_size, seq_len, _ = lstm_output.size()
        query = self.query_vector.repeat(batch_size, 1)
        att_scores = torch.bmm(lstm_output, torch.matmul(query.unsqueeze(1), self.att_weights.unsqueeze(0)).transpose(1, 2))
        att_weights = F.softmax(att_scores, dim=1)
        att_out = torch.bmm(lstm_output.transpose(1, 2), att_weights).squeeze(2)
        return att_out

class A_CRNN(nn.Module):
    def __init__(self, num_of_characters):
        super(A_CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(65536, 128)
        
        self.attention1 = Attention(128)  # Attention before first LSTM
        self.lstm1 = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        
        self.attention2 = Attention(512)  # Attention before second LSTM
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        self.fc2 = nn.Linear(512, num_of_characters)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.attention1(x.unsqueeze(1))
        x, _ = self.lstm1(x)

        x = self.attention2(x.unsqueeze(1))
        x, _ = self.lstm2(x)

        x = self.fc2(x)

        return F.softmax(x, dim=-1)
