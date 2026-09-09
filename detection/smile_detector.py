import cv2
import numpy as np
from typing import List, Dict, Any

class SmileDetector:
    def __init__(self):
        # Load OpenCV Haar cascade classifiers for face and smile detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect_smiles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces and smiles in frame.
        Returns list of detections: [{'class_id': 99, 'label': 'person_smiling', 'confidence': float, 'bbox': [x1, y1, x2, y2]}]
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        smile_detections = []

        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))

            if len(smiles) > 0:
                smile_detections.append({
                    "class_id": 99,
                    "label": "person_smiling",
                    "confidence": 0.92,
                    "bbox": [fx, fy, fx + fw, fy + fh]
                })

        return smile_detections
