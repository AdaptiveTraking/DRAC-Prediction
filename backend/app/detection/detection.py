from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, stream=True, verbose=False, device='0')

        drone_detected = []
        for result in results:
            for box in result.boxes:
                if result.names[box.cls] == 'drone':
                    drone_detected.append({
                        'class': result.names[box.cls],
                        'confidence': box.conf.item(),
                        'box': box.xyxy.cpu().numpy().tolist()
                    })
        return drone_detected

drone_detector = DroneDetector(model_path='backend/models/detection/best.pt')