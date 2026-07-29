from ultralytics import YOLO
from pathlib import Path

class DroneDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=True, verbose=False, device='cpu')

        drone_detected = []
        for result in results:
            for box in result.boxes:
                if result.names[box.cls.item()] == 'drone':
                    drone_detected.append({
                        'class': result.names[box.cls.item()],
                        'confidence': box.conf.item(),
                        'box': box.xyxy.cpu().numpy().flatten().tolist()
                    })
        return drone_detected

if __name__ == '__main__':
    backend_root = Path(__file__).resolve().parents[2]
    model_path = backend_root / "models" / "detection" / "best.pt"
    drone_detector = DroneDetector(model_path=str(model_path))
