"""
Click-to-select single-target tracker.

Click on any detected box on screen to lock onto it. Once locked, the
tracker follows whichever detection is closest to the target's last known
position each frame (same continuity logic as the auto-pick version).
Press 'r' to release the current target and pick a new one by clicking again.
"""

import cv2

class ClickToSelectTracker:
    def __init__(self, max_jump_distance=150):
        self.locked = False
        self.last_cx = None
        self.last_cy = None
        self.max_jump_distance = max_jump_distance
        self.pending_click = None  # (x, y) set by the mouse callback, consumed once per frame

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending_click = (x, y)

    def release(self):
        """Unlock the current target so the next click picks a new one."""
        self.locked = False
        self.last_cx = None
        self.last_cy = None

    def select(self, detections):
        """
        Call once per frame with the list of detections.
        Returns the tracked detection dict, or None if nothing is locked yet.
        """
        # Not locked onto anything yet, check if a click landed inside a detection this frame
        if not self.locked:
            if self.pending_click is not None:
                click_x, click_y = self.pending_click
                self.pending_click = None
                for d in detections:
                    x1, y1, x2, y2 = d['bbox']
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        self.locked = True
                        self.last_cx = (x1 + x2) / 2
                        self.last_cy = (y1 + y2) / 2
                        return d
            return None  # waiting for a click

        if not detections:
            return None

        def dist(d):
            x1, y1, x2, y2 = d['bbox']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return ((cx - self.last_cx) ** 2 + (cy - self.last_cy) ** 2) ** 0.5

        best = min(detections, key=dist)
        if self.max_jump_distance is not None and dist(best) > self.max_jump_distance:
            return None  # target likely lost this frame

        x1, y1, x2, y2 = best['bbox']
        self.last_cx, self.last_cy = (x1 + x2) / 2, (y1 + y2) / 2
        return best
