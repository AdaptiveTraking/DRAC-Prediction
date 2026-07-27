"""
INPUT FEATURES PER FRAME (7): [x1, y1, x2, y2, dt, vx, vy]
    x1,y1,x2,y2 : bounding box corners
    dt          : time since previous frame
    vx, vy      : velocity of the box CENTER
 
OUTPUT: (horizon, 4) — predicted (x1, y1, x2, y2) at each future step.
    NOTE: unlike the original, this no longer predicts velocity — only the box itself.
"""

import torch
import torch.nn as nn
from collections import deque
import numpy as np

class AdaptiveTrajectoryGRU(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1, horizon=5, output_features=6):
        super(AdaptiveTrajectoryGRU, self).__init__()
        self.horizon = horizon
        self.output_features = output_features
 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon * output_features)
 
    def forward(self, x):
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        out = self.fc(last_hidden)
        out = out.view(-1, self.horizon, self.output_features)
        return out


class MultiHorizonLoss(nn.Module):
    def __init__(self, horizon=5, box_weight=1.0, vel_weight=0.5, decay_factor=0.9):
        super(MultiHorizonLoss, self).__init__()

        self.horizon = horizon
        self.box_weight = box_weight
        self.vel_weight = vel_weight
        self.decay_factor = decay_factor
        self.criterion = nn.SmoothL1Loss(reduction='none')
 
    def forward(self, predictions, targets):
        """
        predictions, targets: (batch, horizon, 6) — (x1, y1, x2, y2, vx, vy)
        Box (corners) and velocity are weighted separately, same as the original
        """
        total_loss = 0
 
        for h in range(self.horizon):
            pred_box = predictions[:, h, 0:4]
            pred_vel = predictions[:, h, 4:6]
 
            targ_box = targets[:, h, 0:4]
            targ_vel = targets[:, h, 4:6]
 
            box_loss = self.criterion(pred_box, targ_box).mean()
            vel_loss = self.criterion(pred_vel, targ_vel).mean()
 
            weight = self.decay_factor ** h
            step_loss = weight * (self.box_weight * box_loss + self.vel_weight * vel_loss)
            total_loss += step_loss
 
        return total_loss / self.horizon


class TrajectoryInferencer:
    def __init__(self, model, sequence_length=10):
        self.model = model
        self.model.eval()
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)
        self.last_cx = None
        self.last_cy = None
        self.last_t = None
 
    def update_and_predict(self, x1, y1, x2, y2, t):
        """
        Takes a raw bounding box (x1, y1, x2, y2) and timestamp from detection,
        computes center-velocity, updates the buffer, and returns predictions
        (future corners) once the buffer is full.
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
 
        if self.last_t is None:
            self.last_cx, self.last_cy, self.last_t = cx, cy, t
            # pad with 0 velocity for the first frame
            self.buffer.append([x1, y1, x2, y2, 0.0, 0.0, 0.0])
            return None
 
        dt = t - self.last_t
        if dt > 0:
            vx = (cx - self.last_cx) / dt
            vy = (cy - self.last_cy) / dt
        else:
            vx, vy = 0.0, 0.0
 
        self.buffer.append([x1, y1, x2, y2, dt, vx, vy])
        self.last_cx, self.last_cy, self.last_t = cx, cy, t
 
        if len(self.buffer) == self.sequence_length:
            input_tensor = torch.tensor(list(self.buffer), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predictions = self.model(input_tensor)
            return predictions.squeeze(0).numpy()

        return None

adaptive_trajectoryGRU = AdaptiveTrajectoryGRU()
trajectory_inferencer = TrajectoryInferencer(adaptive_trajectoryGRU)