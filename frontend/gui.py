import base64
import multiprocessing as mp
import time
import tkinter as tk
from pathlib import Path
from queue import Empty, Full
from tkinter import ttk

import cv2 as cv

try:
    from frontend.detection_process import run_detector
    from frontend.tracking import ClickToSelectTracker
except ImportError:
    from detection_process import run_detector
    from tracking import ClickToSelectTracker


class Gui:
    """
    Tkinter camera interface for drone detection and click-to-select tracking.

    The GUI owns three jobs:
    1. Read frames from the camera.
    2. Send the latest frame to a detector process when that process is ready.
    3. Draw the latest boxes, selected target, and text statistics.

    """

    def __init__(self, camera_index=0):
        self.root = tk.Tk()
        self.root.title("Adaptive Tracking")
        self.root.geometry("900x650")
        self.is_closing = False

        # Display settings. refresh_ms controls how often Tkinter asks for the next frame
        # preview_width/height limit the frame size shown in the GUI.
        self.camera_index = camera_index
        self.refresh_ms = 2
        self.preview_width = 860
        self.preview_height = 520

        # camera_image must be stored on self
        self.camera = cv.VideoCapture(self.camera_index)
        self.camera_image = None
        self.current_frame_size = None

        # Keep the queues at size 1. The app should use the newest frame/result,
        # not process a growing backlog of stale frames.
        self.mp_context = mp.get_context("spawn")
        self.frame_queue = self.mp_context.Queue(maxsize=1)
        self.result_queue = self.mp_context.Queue(maxsize=1)
        self.stop_event = self.mp_context.Event()
        self.detector_process = None

        model_path = Path(__file__).resolve().parents[1] / "backend" / "models" / "detection" / "best.pt"
        self.start_detector_process(str(model_path))

        self.tracker = ClickToSelectTracker(max_jump_distance=150)

        # frame_id identifies which displayed frame was sent to detection.
        # detections holds the newest completed worker result. These boxes may
        # lag behind the live camera a little, but the preview stays smooth.
        self.frame_id = 0
        self.detections = []
        self.latest_result = None
        self.detector_error = None


        self.drone_count_var = tk.StringVar(value="Drones present: 0")
        self.drone_positions_var = tk.StringVar(value="Drone positions: None")

        self.build_layout()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-r>", self.release_tracking)
        self.root.bind("<KeyPress-q>", self.close)

        self.update_camera()

    def start_detector_process(self, model_path):
        """
        Start the long-lived detection worker process.

        The worker receives frames through frame_queue and sends completed
        detection results through result_queue. The process is started only once
        because loading the YOLO model repeatedly would be very expensive.
        """
        self.detector_process = self.mp_context.Process(
            target=run_detector,
            args=(
                self.frame_queue,
                self.result_queue,
                self.stop_event,
                model_path,
            ),
            daemon=True,
        )
        self.detector_process.start()

    def build_layout(self):
        """
        Build and place the camera preview and statistics labels.
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.camera_label = tk.Label(
            main_frame,
            text="Starting camera...",
            bg="black",
            fg="white",
            width=100,
            height=30,
        )
        self.camera_label.grid(row=0, column=0, sticky="nsew")

        # The click is bound to the preview widget because the user chooses a
        # target by clicking directly on the displayed detection box.
        self.camera_label.bind("<Button-1>", self.on_camera_click)

        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(
            info_frame,
            textvariable=self.drone_count_var,
            font=("Arial", 14),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            info_frame,
            textvariable=self.drone_positions_var,
            font=("Arial", 14),
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    def update_camera(self):
        """
        Read one camera frame, poll detection results, draw, then schedule again.
        """
        if self.is_closing:
            return

        if not self.camera.isOpened():
            self.camera_label.configure(text="Camera not available", image="")
            return

        success, frame = self.camera.read()

        if success:
            frame = self.resize_frame(frame)
            self.frame_id += 1

            self.send_frame_to_detector(frame)
            self.read_detector_results()

            target = self.tracker.select(self.detections)
            positions = self.draw_detections(frame, self.detections, target)

            self.update_drone_info(len(self.detections), positions, self.latest_result)
            self.show_frame(frame)

        self.root.after(self.refresh_ms, self.update_camera)

    def send_frame_to_detector(self, frame):
        """
        Send the newest display-sized frame to the detector if its queue is free.
        """
        if self.detector_process is None or not self.detector_process.is_alive():
            return

        payload = {
            "frame_id": self.frame_id,
            "timestamp_ms": time.perf_counter() * 1000,
            "frame": frame.copy(),
        }

        try:
            self.frame_queue.put_nowait(payload)
        except Full:
            pass

    def read_detector_results(self):
        """
        Read all available worker results without blocking the GUI.
        """
        while True:
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                break

            self.latest_result = result
            self.detector_error = result.get("error")
            self.detections = result.get("detections", [])

    def on_camera_click(self, event):
        """
        Convert a click on the Tkinter preview widget into frame coordinates.
        """
        if self.current_frame_size is None:
            return

        frame_width, frame_height = self.current_frame_size
        widget_width = self.camera_label.winfo_width()
        widget_height = self.camera_label.winfo_height()

        offset_x = max((widget_width - frame_width) // 2, 0)
        offset_y = max((widget_height - frame_height) // 2, 0)

        click_x = event.x - offset_x
        click_y = event.y - offset_y

        if 0 <= click_x < frame_width and 0 <= click_y < frame_height:
            self.tracker.set_click(click_x, click_y)

    def release_tracking(self, event=None):
        """
        Release the currently selected target.

        The optional event argument lets this method be used both as a normal
        method call and as a Tkinter key binding callback.
        """
        self.tracker.release()

    def get_bbox(self, detection):
        return detection.get("bbox", detection.get("box"))

    def draw_detections(self, frame, detections, target):
        """
        Draw all detection boxes and emphasize the selected target.
        """
        positions = []

        for detection in detections:
            x1, y1, x2, y2 = self.get_bbox(detection)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            positions.append((center_x, center_y))

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0), 1)
            cv.circle(frame, (center_x, center_y), 3, (0, 150, 0), -1)

        if target is None:
            # When nothing is locked, the overlay tells the user what action the
            # interface expects next without changing the statistics labels.
            cv.putText(
                frame,
                "Click a box to track",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            return positions

        x1, y1, x2, y2 = self.get_bbox(target)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        confidence = target.get("confidence", 0)
        class_name = target.get("class", "drone")

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(
            frame,
            f"{class_name} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        return positions

    def resize_frame(self, frame):
        """
        Resize the camera frame to fit inside the preview area.
        """
        height, width = frame.shape[:2]

        scale = min(
            self.preview_width / width,
            self.preview_height / height,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        interpolation = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
        resized = cv.resize(frame, (new_width, new_height), interpolation=interpolation)

        self.current_frame_size = (new_width, new_height)

        return resized

    def show_frame(self, frame):
        """
        Convert an OpenCV frame into a Tkinter image and display it.

        """
        success, encoded_image = cv.imencode(".png", frame)

        if not success:
            return

        image_data = base64.b64encode(encoded_image).decode("ascii")
        self.camera_image = tk.PhotoImage(data=image_data, format="png")

        self.camera_label.configure(image=self.camera_image, text="")

    def update_drone_info(self, count=0, positions=None, latest_result=None):
        """
        Update the text statistics under the camera preview.

        count is the number of currently cached detections. positions contains
        the center point of each detection box, which is easier to read than four
        bounding-box corner values for every drone.
        """
        if positions is None:
            positions = []

        inference_text = ""

        if latest_result is not None:
            inference_ms = latest_result.get("stats", {}).get("inference_ms")

            if inference_ms is not None:
                inference_text = f" | Detection: {inference_ms:.0f} ms"

        if self.detector_error is not None:
            inference_text = f" | {self.detector_error}"

        self.drone_count_var.set(f"Drones present: {count}{inference_text}")

        if not positions:
            self.drone_positions_var.set("Drone positions: None")
            return

        formatted_positions = ", ".join(
            f"({int(x)}, {int(y)})" for x, y in positions
        )
        self.drone_positions_var.set(f"Drone positions: {formatted_positions}")

    def close(self, event=None):
        """
        Stop the worker process, release the camera, and close the window.

        """
        if self.is_closing:
            return

        self.is_closing = True
        self.stop_event.set()

        try:
            self.frame_queue.put_nowait(None)
        except Full:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass

            try:
                self.frame_queue.put_nowait(None)
            except Full:
                pass

        if self.detector_process is not None and self.detector_process.is_alive():
            self.detector_process.join(timeout=1.0)

            if self.detector_process.is_alive():
                self.detector_process.terminate()
                self.detector_process.join(timeout=1.0)

        if self.camera.isOpened():
            self.camera.release()

        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    gui = Gui()
    gui.run()
