import cv2
import numpy as np
from typing import Tuple, List

class MotionDetector:
    def __init__(self, threshold: int = 25, min_contour_area: int = 500):
        self.threshold = threshold
        self.min_contour_area = min_contour_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def detect(self, frame: np.ndarray) -> Tuple[bool, List[List[int]], float]:
        if frame is None:
            return False, [], 0.0

        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        total_motion_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append([x, y, w, h])
                total_motion_area += area

        frame_area = frame.shape[0] * frame.shape[1]
        motion_ratio = min(1.0, total_motion_area / frame_area)
        has_motion = len(bboxes) > 0

        return has_motion, bboxes, motion_ratio
