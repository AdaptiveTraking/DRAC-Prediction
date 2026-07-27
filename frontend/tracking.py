import cv2


class ClickToSelectTracker:
    """
    Keeps track of one user-selected drone detection across frames.

    The detector may return many boxes per frame, but the prediction step should
    normally follow one target. This class separates that selection logic from
    the GUI: the GUI only sends clicks and detections, while the tracker decides
    which detection is currently the active target.
    """

    def __init__(self, max_jump_distance=150):
        """
        Create an unlocked tracker.

        max_jump_distance limits how far the selected target is allowed to move
        between detection updates. This prevents the tracker from jumping to a
        different drone if the original target disappears or the detector misses
        it for a frame.
        """
        self.locked = False
        self.last_cx = None
        self.last_cy = None
        self.max_jump_distance = max_jump_distance
        self.pending_click = None

    def set_click(self, x, y):
        """
        Store a click location until the next call to select().

        The GUI receives mouse events immediately, but detections are processed
        frame by frame. Keeping the click as pending lets select() compare the
        click against the latest available detection boxes in one place.
        """
        self.pending_click = (x, y)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.set_click(x, y)

    def release(self):
        """
        Unlock the current target.

        After release, select() waits for a new click before returning a target.
        This is useful when the user wants to stop tracking one drone and choose
        another one without restarting the application.
        """
        self.locked = False
        self.last_cx = None
        self.last_cy = None
        self.pending_click = None

    def get_bbox(self, detection):
        return detection.get("bbox", detection.get("box"))

    def select(self, detections):
        """
        Pick or update the active target from the current frame's detections.

        If the tracker is not locked, a pending click is tested against every
        detection box. The first box containing the click becomes the target.

        If the tracker is already locked, the target is updated by choosing the
        detection whose center is closest to the last known target center. This
        gives simple frame-to-frame continuity without running a full tracker.
        """
        if not self.locked:
            if self.pending_click is not None:
                click_x, click_y = self.pending_click
                self.pending_click = None

                for detection in detections:
                    x1, y1, x2, y2 = self.get_bbox(detection)

                    # A click only selects a detection if it lands inside the box.
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        self.locked = True
                        self.last_cx = (x1 + x2) / 2
                        self.last_cy = (y1 + y2) / 2
                        return detection

            return None

        if not detections:
            return None

        def distance(detection):
            """
            Measure how far a detection is from the last target position.

            The center point is used because it is stable even if box width and
            height change slightly between detections.
            """
            x1, y1, x2, y2 = self.get_bbox(detection)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return ((cx - self.last_cx) ** 2 + (cy - self.last_cy) ** 2) ** 0.5

        best = min(detections, key=distance)

        # If the nearest detection is too far away, treat the target as missing
        # for this frame instead of accidentally switching to another drone.
        if self.max_jump_distance is not None and distance(best) > self.max_jump_distance:
            return None

        x1, y1, x2, y2 = self.get_bbox(best)
        self.last_cx = (x1 + x2) / 2
        self.last_cy = (y1 + y2) / 2

        return best
