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



    class MultiHorizonLoss(nn.Module):
     def __init__(self, horizon=5, pos_weight=1.0, vel_weight=0.5, decay_factor=0.9):
        super(MultiHorizonLoss, self).__init__()
        self.horizon = horizon
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.decay_factor = decay_factor
        self.criterion = nn.SmoothL1Loss(reduction='none')

     def forward(self, predictions, targets):
        total_loss = 0
        
        for h in range(self.horizon):
            pred_pos = predictions[:, h, 0:2]
            pred_vel = predictions[:, h, 2:4]
            
            targ_pos = targets[:, h, 0:2]
            targ_vel = targets[:, h, 2:4]
            
            pos_loss = self.criterion(pred_pos, targ_pos).mean()
            vel_loss = self.criterion(pred_vel, targ_vel).mean()
            
            weight = self.decay_factor ** h
            
            step_loss = weight * (self.pos_weight * pos_loss + self.vel_weight * vel_loss)
            total_loss += step_loss
            
        return total_loss / self.horizon