import cv2
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", imgsz: int = 640):
        self.model_name = model_name
        self.imgsz = imgsz
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        if frame is None or frame.size == 0:
            return []

        results = self.model(frame, imgsz=self.imgsz, verbose=False)
        detections = []

        if not results:
            return detections

        for res in results:
            boxes = res.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id] if hasattr(self.model, 'names') and cls_id in self.model.names else str(cls_id)
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": cls_id,
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": xyxy
                })

        return detections
