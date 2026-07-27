import numpy as np
import torch
from gru import (
    GRUPredictor, SEQ_LEN, NUM_FEATURES, HIDDEN_SIZE,
    NUM_LAYERS, HORIZONS, BBOX_IDX, to_delta_time,
)

class TrajectoryPredictor:
    def __init__(self, model_path, norm_stats_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_trained_model(model_path)
        self.feat_mean, self.feat_std = self.__load_norm_stats(norm_stats_path)

    def __load_trained_model(self, path):
        model = GRUPredictor(
            input_size=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=len(HORIZONS) * 4,
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def __load_norm_stats(self, path):
        stats = np.load(path)  # shape (2, 7)
        return stats[0], stats[1]

    def predict_future_bboxes(self, last_seq_raw):
        """
        last_seq_raw: 
            the most recent SEQ_LEN frames, shape (SEQ_LEN, 7),
            with RAW timestamp in column 0 (not yet delta'd/normalized).
        
        Returns: dict mapping horizon -> [x1, y1, x2, y2]
        """
        seq = to_delta_time(np.array(last_seq_raw, dtype=np.float32))
        seq_norm = (seq - self.feat_mean) / self.feat_std
        x = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy()[0]

        bbox_mean = self.feat_mean[BBOX_IDX]
        bbox_std = self.feat_std[BBOX_IDX]
        pred_real = pred_norm * bbox_std + bbox_mean

        return {h: pred_real[i].tolist() for i, h in enumerate(HORIZONS)}
 
trajectory_predictor = TrajectoryPredictor(
    model_path='backend/models/prediction/gru_model.pt',
    norm_stats_path='backend/models/prediction/norm_stats.npy'
)