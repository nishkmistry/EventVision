import os
import cv2
import logging
from typing import List, Dict, Any

logger = logging.getLogger("SmileDetector")

class SmileDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier()
        self.smile_cascade = cv2.CascadeClassifier()
        self._load_cascades()

    def _load_cascades(self):
        # Candidate paths for face cascade
        face_paths = [
            os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml') if hasattr(cv2, 'data') else '',
            'haarcascade_frontalface_default.xml'
        ]
        # Candidate paths for smile cascade
        smile_paths = [
            os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml') if hasattr(cv2, 'data') else '',
            'haarcascade_smile.xml'
        ]

        for path in face_paths:
            if path and os.path.exists(path):
                if self.face_cascade.load(path):
                    break

        for path in smile_paths:
            if path and os.path.exists(path):
                if self.smile_cascade.load(path):
                    break

        if self.face_cascade.empty():
            logger.warning("Face cascade failed to load. Smile detection disabled gracefully.")
        if self.smile_cascade.empty():
            logger.warning("Smile cascade failed to load. Smile detection disabled gracefully.")

    def detect_smiles(self, frame) -> List[Dict[str, Any]]:
        """
        Detect faces and smiles in frame.
        Guarded against empty cascades to prevent cv2.error Assertion failed !empty() crashes.
        """
        if frame is None or frame.size == 0:
            return []

        if self.face_cascade.empty() or self.smile_cascade.empty():
            return []

        try:
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
        except Exception as e:
            logger.error(f"Error during smile detection: {e}")
            return []
