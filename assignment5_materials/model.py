import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HEIGHT, WIDTH, lstm_seq_length

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))

        return self.head(x)


class DQN_LSTM(nn.Module):
    def __init__(self, action_size):
        super(DQN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(256, action_size)
        # Define an LSTM layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1)

    def forward(self, x, hidden = None):
        # x: (batch_size, seq_len, height, width)
        N = x.size(0)
        seq_len = x.size(1)

        # create placeholder for LSTM input (seq_len, N, input_size)
        lstm_input = torch.zeros(seq_len, N, 512).to(x.device)

        for i in range(seq_len):
            s_i = x[:, i, :, :].unsqueeze(1)
            s_i = F.relu(self.bn1(self.conv1(s_i)))
            s_i = F.relu(self.bn2(self.conv2(s_i)))
            s_i = F.relu(self.bn3(self.conv3(s_i)))
            s_i = F.relu(self.fc(s_i.view(s_i.size(0), -1))) # (N, 512)
            lstm_input[i] = s_i

        # Pass the state through an LSTM
        ### CODE ###
        _, (h_n, c_n) = self.lstm(lstm_input, hidden) # ((1, N, 256), (1, N, 256))

        return self.head(h_n.squeeze()), (h_n, c_n)