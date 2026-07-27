from fastapi import APIRouter
from backend.app.detection.detection import drone_detector
from backend.app.prediction.trajectory_model import trajectory_inferencer

router = APIRouter()

@router.post("/detect")
async def detect_objects_endpoint(image: bytes):
    """
    Endpoint to detect objects in an image using the specified detection model.
    """
    results = drone_detector.detect(image)
    return {"results": results}

class PredictionRequest:
    def __init__(self, x1, y1, x2, y2, t):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.t = t

@router.get("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint to predict future positions based on the last known bounding box and timestamp.
    """
    predictions = trajectory_inferencer.update_and_predict(request.x1, request.y1, request.x2, request.y2, request.t)
    if predictions is not None:
        return {"predictions": predictions.tolist()}
    else:
        return {"message": "Insufficient data for prediction. Please provide more frames."}