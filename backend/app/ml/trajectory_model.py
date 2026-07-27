import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np

class AdaptiveTrajectoryGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=1, horizon=5, output_features=4):
        super(AdaptiveTrajectoryGRU, self).__init__()
        self.horizon = horizon
        self.output_features = output_features
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, horizon * output_features)

    def forward(self, x):

        gru_out, hidden = self.gru(x)

        last_hidden = hidden[-1] 
        out = self.fc(last_hidden)
        
        out = out.view(-1, self.horizon, self.output_features)
        return out

    