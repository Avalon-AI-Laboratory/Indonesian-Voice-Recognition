import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_of_characters):
        super(A_CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(256)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(512, 512)
        
        self.attention1 = Attention(512)  # Attention before first LSTM
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
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
        x = x.view(1, 256, 512)
        x = self.fc1(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc2(x)

        return F.softmax(x, dim = -1)
