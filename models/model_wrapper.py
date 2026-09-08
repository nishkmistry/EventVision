import time
from typing import Dict, Any
from ultralytics import YOLO

class BaseModelWrapper:
    def __init__(self, name: str, model_path: str, imgsz: int = 640):
        self.name = name
        self.model_path = model_path
        self.imgsz = imgsz
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def infer(self, image_crop_or_frame, conf_threshold: float = 0.25) -> Dict[str, Any]:
        self.load_model()
        start = time.time()
        results = self.model(image_crop_or_frame, imgsz=self.imgsz, verbose=False)
        latency = (time.time() - start) * 1000.0

        detections = []
        if results:
            for res in results:
                if res.boxes is not None:
                    for box in res.boxes:
                        c = float(box.conf[0])
                        if c >= conf_threshold:
                            cls_id = int(box.cls[0])
                            label = self.model.names[cls_id] if hasattr(self.model, 'names') and cls_id in self.model.names else str(cls_id)
                            detections.append({
                                "class_id": cls_id,
                                "label": label,
                                "confidence": round(c, 4),
                                "bbox": [int(v) for v in box.xyxy[0].tolist()]
                            })

        return {
            "model_name": self.name,
            "latency_ms": round(latency, 2),
            "detections": detections
        }

class LightweightModel(BaseModelWrapper):
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 320):
        super().__init__("Lightweight (YOLOv8n)", model_path, imgsz)

class StandardModel(BaseModelWrapper):
    def __init__(self, model_path: str = "yolov8s.pt", imgsz: int = 640):
        super().__init__("Standard (YOLOv8s)", model_path, imgsz)

class HighAccuracyModel(BaseModelWrapper):
    def __init__(self, model_path: str = "yolov8m.pt", imgsz: int = 640):
        super().__init__("High Accuracy (YOLOv8m)", model_path, imgsz)
