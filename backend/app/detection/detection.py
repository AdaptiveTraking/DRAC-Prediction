import torch
from ultralytics import YOLO
from pathlib import Path

class DroneDetector:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "0" if torch.cuda.is_available() else "-1"

        self.device = device
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=True, verbose=False, device = self.model.device)

        drone_detected = []
        for result in results:
            for box in result.boxes:
                if result.names[box.cls.item()] == 'drone':
                    drone_detected.append({
                        'class': result.names[box.cls.item()],
                        'confidence': box.conf.item(),
                        'bbox': box.xyxy.cpu().numpy().flatten().tolist()
                    })
        return drone_detected

backend_root = Path(__file__).resolve().parents[2]
model_path = backend_root / "models" / "detection" / "best.pt"
drone_detector = DroneDetector(model_path=str(model_path), device='0') # we will run always on GPU
