"""

Given the last SEQ_LEN frames of an object's tracked state, predicts its bounding box at multiple future horizons: +1, +2, +4, +8, +16 frames ahead.

INPUT FEATURES PER FRAME (7):
    0: delta_t     (time since previous frame, NOT raw timestamp, see note below)
    1: x1          (bbox bottom-left x)
    2: y1          (bbox bottom-left y)
    3: x2          (bbox top-right x)
    4: y2          (bbox top-right y)
    5: speed
    6: direction
 
OUTPUT: for each horizon in HORIZONS, predicted (x1, y1, x2, y2)
    shape: (batch, num_horizons, 4)

Why delta_t instead of raw timestamp?
Row epoch timestamp are huge numbers, and the GRU will have a hard time learning from them. Instead, we use the time difference between frames, which is a much smaller number and more relevant to the prediction task.

"""

import numpy as np
# ! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
import torch
import pandas as pd
import torch.nn as nn
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import Model

from torch.utils.data import DataLoader, Dataset

SEQ_LEN = 20                # how many past frames to consider for prediction
"""
    for NUM_FEATURES we will use 7 features:
    - delta_t
    - x1 position - so this is the bottom left corner of the bounding box
    - y1 position 
    - x2 position - so this is the top right corner of the bounding box
    - y2 position
    - speed
    - direction
"""
NUM_FEATURES = 7            # how many variables per frame
HORIZONS = [1, 2, 4, 8, 16] # frames-ahead to predict
BBOX_IDX = [1, 2, 3, 4]     # indices of x1,y1,x2,y2 within the 7 features
 
HIDDEN_SIZE = 64            # number of features in the hidden state
NUM_LAYERS = 2              # stacked GRU layers
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path):
    """
    Reads the real tracking data, sorts by timestamp, and converts
    timestamp -> delta_t. Returns a plain numpy array so downstream
    indexing (make_sequences, normalize_data) stays simple and unambiguous.
 
    Return shape: (num_timesteps, 7) columns:
        [delta_t, x1, y1, x2, y2, speed, direction]
    """
    df = pd.read_csv(csv_path)
    expected_columns = ['timestamp', 'x1', 'y1', 'x2', 'y2', 'speed', 'direction']
    # Ensure the dataframe has the expected columns
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"CSV file must contain the following columns: {expected_columns}")
 
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['delta_t'] = df['timestamp'].diff().fillna(0)
 
    return df[['delta_t', 'x1', 'y1', 'x2', 'y2', 'speed', 'direction']].to_numpy(dtype=np.float32)
 
 
def to_delta_time(data: np.ndarray):
    """Converts column 0 from raw timestamp to delta since previous frame."""
    data = data.copy()
    dt = np.diff(data[:, 0], prepend=data[0, 0])
    dt[0] = dt[1] if len(dt) > 1 else 0.0
    data[:, 0] = dt
    return data
 
 
def normalize_data(raw: np.ndarray):
    """
    Computes per-feature mean/std for z-score normalization.
    `raw` is already delta_t-converted and sorted by time (done in load_data).
    Returns: raw (unchanged), data_norm (z-scored), feat_mean, feat_std
    """
    feat_mean = raw.mean(axis=0)
    feat_std = raw.std(axis=0) + 1e-8
    data_norm = (raw - feat_mean) / feat_std
    return raw, data_norm, feat_mean, feat_std
 
 
def make_sequences(data_norm: np.ndarray, data_raw: np.ndarray, seq_len: int, horizons: list):
    """
    X comes from the NORMALIZED data (model input).
    y comes from the RAW data (so it only gets normalized once, via bbox_mean/std,
    not twice — this was a bug in an earlier version of this script).
 
      X: (N, seq_len, 7)          -- past window, normalized
      y: (N, len(horizons), 4)    -- future bbox at each horizon, RAW scale
    """
    max_h = max(horizons)
    X, y = [], []
    for i in range(len(data_norm) - seq_len - max_h + 1):
        X.append(data_norm[i:i + seq_len])
        last_idx = i + seq_len - 1
        targets = [data_raw[last_idx + h, BBOX_IDX] for h in horizons]
        y.append(targets)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y
 
 
class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GRUPredictor(nn.Module):
    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=len(HORIZONS) * 4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_horizons = len(HORIZONS)
 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        out, _ = self.gru(x)               
        out = self.fc(out[:, -1, :])         
        return out.view(-1, self.num_horizons, 4)
 
 
# TRAIN
def train_model(model, train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
        val_loss /= len(val_loader.dataset)
 
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
 
 
if __name__ == "__main__":
    raw = load_data('backend/models/prediction/drone_tracking_data.csv')
    data_raw, data_norm, feat_mean, feat_std = normalize_data(raw)
 
    X, y_raw = make_sequences(data_norm, data_raw, SEQ_LEN, HORIZONS)
 
    bbox_mean = feat_mean[BBOX_IDX]
    bbox_std = feat_std[BBOX_IDX]
    y = (y_raw - bbox_mean) / bbox_std   # normalized exactly once
 
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
 
    train_dataset = TrajectoryDataset(X_train, y_train)
    val_dataset = TrajectoryDataset(X_val, y_val)
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
 
    model = GRUPredictor().to(DEVICE)
    train_model(model, train_loader, val_loader)
 
    torch.save(model.state_dict(), "gru_model.pt")
    np.save("norm_stats.npy", np.stack([feat_mean, feat_std]))
    print("\nModel saved to gru_model.pt")
    print("Per-feature normalization stats saved to norm_stats.npy")
 