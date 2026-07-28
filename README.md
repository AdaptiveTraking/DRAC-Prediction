# DRAC-Prediction
Dynamic Real-time Adaptive Coordinates for Drone Trajectory Prediction

This is a repository containing an app that detects and tracks drones from a video stream, and predicts their future trajectory using a GRU-based model. The app is embedded so no API calls are done.

### To run you need the models!
Download the trained models from [this Google Drive](https://drive.google.com/drive/folders/1VB1GTvkngcJbK1OqWzLU-DrBS6yGDbos?usp=sharing) and place them in the `backend/models/` directory.

### Run the app
```bash
cd backend
python -m app.main
```

### Demo