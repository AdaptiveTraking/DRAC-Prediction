"""
Background process entry point for CPU drone detection.
"""

import sys
import time
from pathlib import Path
from queue import Empty, Full

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_detector(frame_queue, result_queue, stop_event, model_path, torch_threads=2):
    """
    Load the detector once, then process the newest frames sent by the GUI.

    frame_queue receives dictionaries from the main process:
        {
            "frame_id": int,
            "timestamp_ms": float,
            "frame": numpy.ndarray,
        }

    result_queue sends dictionaries back to the GUI:
        {
            "frame_id": int,
            "timestamp_ms": float,
            "detections": list[dict],
            "stats": dict,
        }

    Both queues should have maxsize=1 (drop old info and replace it basically)
    """
    limit_torch_threads(torch_threads)

    try:
        from backend.app.detection.detection import DroneDetector

        detector = DroneDetector(model_path=model_path)
    except Exception as exc:
        put_latest(
            result_queue,
            {
                "frame_id": None,
                "timestamp_ms": None,
                "detections": [],
                "stats": {
                    "count": 0,
                    "positions": [],
                    "inference_ms": 0.0,
                },
                "error": f"Could not load detector: {exc}",
            },
        )
        return

    while not stop_event.is_set():
        try:
            payload = frame_queue.get(timeout=0.1)
        except Empty:
            continue

        if payload is None:
            break

        frame_id = payload["frame_id"]
        timestamp_ms = payload["timestamp_ms"]
        frame = payload["frame"]

        start = time.perf_counter()

        try:
            raw_detections = detector.detect(frame)
            detections = normalize_detections(raw_detections)
            error = None
        except Exception as exc:
            detections = []
            error = f"Detection failed: {exc}"

        inference_ms = (time.perf_counter() - start) * 1000
        positions = [detection["center"] for detection in detections]

        result = {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": detections,
            "stats": {
                "count": len(detections),
                "positions": positions,
                "inference_ms": inference_ms,
            },
        }

        if error is not None:
            result["error"] = error

        put_latest(result_queue, result)


def limit_torch_threads(torch_threads):
    """
    Reserve some CPU time for the GUI while the detector process runs.
    """
    if torch_threads is None:
        return

    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:
        pass


def normalize_detections(raw_detections):
    """
    Convert detector output
    """
    detections = []

    for raw_detection in raw_detections:
        bbox = raw_detection.get("bbox", raw_detection.get("box"))

        if bbox is None:
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detections.append(
            {
                "class": raw_detection.get("class", "drone"),
                "confidence": float(raw_detection.get("confidence", 0.0)),
                "bbox": [x1, y1, x2, y2],
                "center": [center_x, center_y],
            }
        )

    return detections


def put_latest(result_queue, result):
    """
    Put a result into a maxsize=1 queue

    Gui only needs the latest boxes to show (for latest frames sent) so, if the pipe towards Gui (main) already contains a
    result, but the detector sends another, drop the older one and replace it
    """
    try:
        result_queue.put_nowait(result)
        return
    except Full:
        pass

    try:
        result_queue.get_nowait()
    except Empty:
        pass

    try:
        result_queue.put_nowait(result)
    except Full:
        pass
