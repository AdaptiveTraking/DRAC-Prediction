import numpy as np
import torch
from gru import (
    GRUPredictor, SEQ_LEN, NUM_FEATURES, HIDDEN_SIZE,
    NUM_LAYERS, HORIZONS, BBOX_IDX, to_delta_time,
)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(path="backend/models/prediction/gru_model.pt"):
    model = GRUPredictor(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(HORIZONS) * 4,
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_future_bboxes(model, last_seq_raw, feat_mean, feat_std):
    """
    last_seq_raw: 
        the most recent SEQ_LEN frames, shape (SEQ_LEN, 7),
        with RAW timestamp in column 0 (not yet delta'd/normalized).
    feat_mean, feat_std: per-feature stats saved during training (norm_stats.npy).
 
    Returns: dict mapping horizon -> [x1, y1, x2, y2]
    """
    seq = to_delta_time(np.array(last_seq_raw, dtype=np.float32))
    seq_norm = (seq - feat_mean) / feat_std
    x = torch.from_numpy(seq_norm).unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, 7)
 
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()[0]  # (num_horizons, 4)
 
    bbox_mean = feat_mean[BBOX_IDX]
    bbox_std = feat_std[BBOX_IDX]
    pred_real = pred_norm * bbox_std + bbox_mean
 
    return {h: pred_real[i].tolist() for i, h in enumerate(HORIZONS)}
 
 
if __name__ == "__main__":
    model = load_trained_model("backend/models/prediction/gru_model.pt")
    stats = np.load("backend/models/prediction/norm_stats.npy")  # shape (2, 7)
    feat_mean, feat_std = stats[0], stats[1]
 
    # dummy example: replace with your real last 20 tracked frames
    dummy_seq = np.random.randn(SEQ_LEN, NUM_FEATURES).astype(np.float32)
    dummy_seq[:, 0] = np.cumsum(np.abs(dummy_seq[:, 0])) / 30.0  # fake increasing timestamps
 
    predictions = predict_future_bboxes(model, dummy_seq, feat_mean, feat_std)
 
    print("Predicted future bounding boxes:")
    for h, bbox in predictions.items():
        print(f"  +{h:2d} frames: x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}")