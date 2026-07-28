"""
INPUT FEATURES PER FRAME (5): [x, y, dt, vx, vy]
    x, y   : drone center position
    dt     : time since previous frame
    vx, vy : center velocity

OUTPUT: (horizon, 4) predicted (x, y, vx, vy) at each future step.
"""

import torch
import torch.nn as nn
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

    @classmethod
    def load(cls, weights_path, input_size=5, hidden_size=64, num_layers=1,
              horizon=5, output_features=4, device=None):
        """
        Usage:
            model = AdaptiveTrajectoryGRU.load("adaptive_trajectory_gru.pt")
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            horizon=horizon, output_features=output_features,
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model


class MultiHorizonLoss(nn.Module):
    def __init__(self, horizon=5, pos_weight=1.0, vel_weight=0.5, decay_factor=0.9):
        super(MultiHorizonLoss, self).__init__()
        self.horizon = horizon
        self.pos_weight = pos_weight
        self.vel_weight = vel_weight
        self.decay_factor = decay_factor
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        """
        predictions, targets: (batch, horizon, 4) — (x, y, vx, vy)
        """
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


class TrajectoryInferencer:
    def __init__(self, model, sequence_length=10, feat_mean=None, feat_std=None):
        """
        feat_mean, feat_std: per-feature normalization stats from training
            (norm_stats.npy, shape (2, 5) for [x, y, dt, vx, vy]).
            REQUIRED if the model was trained on normalized data (it was) —
            without these, predictions will be meaningless, since the model
            never saw raw-pixel-scale inputs during training.
        """
        self.model = model
        self.model.eval()
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)
        self.last_cx = None
        self.last_cy = None
        self.last_t = None

        self.feat_mean = feat_mean
        self.feat_std = feat_std
        # target stats correspond to [x, y, vx, vy] -> feature indices [0, 1, 3, 4]
        if feat_mean is not None:
            self.target_mean = feat_mean[[0, 1, 3, 4]]
            self.target_std = feat_std[[0, 1, 3, 4]]

    def update_and_predict(self, x1, y1, x2, y2, t):
        """
        Takes a raw bounding box (x1, y1, x2, y2) and timestamp from
        detection. Computes the center point + velocity, updates the
        buffer, and returns predictions once the buffer is full.
        Returns an array of shape (horizon, 4): (x, y, vx, vy) per step,
        in the SAME real-world scale as the input (already un-normalized).
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if self.last_t is None:
            self.last_cx, self.last_cy, self.last_t = cx, cy, t
            self.buffer.append([cx, cy, 0.0, 0.0, 0.0])  # pad with 0 velocity, first frame
            return None

        dt = t - self.last_t
        if dt > 0:
            vx = (cx - self.last_cx) / dt
            vy = (cy - self.last_cy) / dt
        else:
            vx, vy = 0.0, 0.0

        self.buffer.append([cx, cy, dt, vx, vy])
        self.last_cx, self.last_cy, self.last_t = cx, cy, t

        if len(self.buffer) == self.sequence_length:
            raw_seq = np.array(self.buffer, dtype=np.float32)

            if self.feat_mean is not None:
                seq = (raw_seq - self.feat_mean) / self.feat_std
            else:
                seq = raw_seq

            device = next(self.model.parameters()).device 
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_norm = self.model(input_tensor).squeeze(0).cpu().numpy()

            if self.feat_mean is not None:
                return pred_norm * self.target_std + self.target_mean
            return pred_norm

        return None

model_gru = AdaptiveTrajectoryGRU.load(weights_path="models/prediction/adaptive_trajectory_gru.pt")

stats = np.load("models/prediction/norm_stats.npy")
feat_mean, feat_std = stats[0], stats[1]

trajectory_inferencer = TrajectoryInferencer(model=model_gru, feat_mean=feat_mean, feat_std=feat_std)