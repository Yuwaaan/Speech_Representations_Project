import torch
import torch.nn as nn
import numpy as np

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class APCModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=512, layers=3):
        super(APCModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 40)
    def forward(self, input):
        out, (hidden_state, hidden_cell) = self.lstm(input)
        out = self.fc(out)
        return out

net = APCModel().to(device)
print(net)