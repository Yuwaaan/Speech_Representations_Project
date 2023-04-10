import numpy as np
from apc import APCModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import numpy as np
from apc import APCModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class FrameClassifier(nn.Module):
    def __init__(self, n_class, pretrain_path):
        super(FrameClassifier, self).__init__()
        # load pretained model parameters
        APC_model = APCModel()
        checkpoint = torch.load(pretrain_path, map_location=device)
        APC_model.load_state_dict(torch.load(pretrain_path), strict=False)
        pretrain_optimizer = torch.optim.Adam(APC_model.parameters(), lr=0.001)
        
        # load lstms
        self.pretrain_model = nn.Sequential(*list(APC_model.children())[:-1])  
        
        for p in APC_model.parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(512, n_class)
        
    def forward(self, mat):
#         [1, 652, 40]
#         mat = torch.squeeze(mat, dim=0)
        mat, (hidden_state, hidden_cell) = self.pretrain_model(mat)   
#         mat = mat[0]
#         mat, (hidden_state, hidden_cell) = self.lstm(mat)
        mat = torch.squeeze(mat, dim=0)
        out = self.fc(mat)
#         out = torch.nn.functional.softmax(out,1)
        return out


# class FrameClassifier(nn.Module):
#     def __init__(self, n_class, pretrain_path):
#         super(FrameClassifier, self).__init__()
#         # load pretained model parameters
#         APC_model = APCModel()
#         checkpoint = torch.load(pretrain_path, map_location=device)
#         APC_model.load_state_dict(torch.load(pretrain_path), strict=False)
#         pretrain_optimizer = torch.optim.Adam(APC_model.parameters(), lr=0.001)
        
#         # load lstms
#         self.pretrain_model = nn.Sequential(*list(APC_model.children())[:-1])  
        
#         for p in APC_model.parameters():
#             p.requires_grad = False
#         self.fc = nn.Linear(512, n_class)
        
#     def forward(self, mat):
# #         [1, 652, 40]
# #         mat = torch.squeeze(mat, dim=0)
#         mat, (hidden_state, hidden_cell) = self.pretrain_model(mat)   # [652, 40]
#         mat = mat[0]
#         out = self.fc(mat)
#         out = torch.nn.functional.softmax(out,0)
#         return out

# net = FrameClassifier(n_class, pretrain_path).to(device)
# print(net)