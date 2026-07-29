import cv2
from app.detection.detection import drone_detector
from app.tracking.tracking import ClickToSelectTracker
from app.prediction.trajectory_model import trajectory_inferencer

cap = cv2.VideoCapture("./test/test2.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_name = "Drone Tracking"
cv2.namedWindow(window_name)
tracker = ClickToSelectTracker(max_jump_distance=150)
cv2.setMouseCallback(window_name, tracker.mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = drone_detector.detect(frame)

    # draw every detection faintly so the user can see what's clickable
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 150, 0), 1)

    target = tracker.select(detections)

    if target is None:
        cv2.putText(frame, "Click a box to track", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    else:
        bbox = target['bbox']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"{target['class']} {target['confidence']:.2f}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        future_positions = trajectory_inferencer.update_and_predict(
            bbox[0], bbox[1], bbox[2], bbox[3], t=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        )
        print(future_positions)

        if future_positions is not None:  # each row: (x, y, vx, vy)
            HORIZONS = [1, 2, 4, 8, 16]
            for h, (x, y, vx, vy) in zip(HORIZONS, future_positions):
                center = (int(x), int(y))
                cv2.circle(frame, center, 6, (255, 0, 0), -1)
                cv2.putText(frame, f"+{h}", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        tracker.release()  # let the user click a different target

cap.release()
cv2.destroyAllWindows()