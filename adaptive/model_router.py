from models.model_wrapper import LightweightModel, StandardModel, HighAccuracyModel
from typing import Dict, Any

class ModelRouter:
    def __init__(self, models_cfg: Dict[str, Any] = None):
        cfg = models_cfg or {}
        lw_path = cfg.get("lightweight", {}).get("name", "yolov8n.pt")
        lw_sz = cfg.get("lightweight", {}).get("imgsz", 320)

        st_path = cfg.get("standard", {}).get("name", "yolov8s.pt")
        st_sz = cfg.get("standard", {}).get("imgsz", 640)

        ha_path = cfg.get("high_accuracy", {}).get("name", "yolov8m.pt")
        ha_sz = cfg.get("high_accuracy", {}).get("imgsz", 640)

        self.lightweight = LightweightModel(lw_path, lw_sz)
        self.standard = StandardModel(st_path, st_sz)
        self.high_accuracy = HighAccuracyModel(ha_path, ha_sz)

    def route_and_process(self, policy: str, frame_or_crop) -> Dict[str, Any]:
        if policy == "SKIP":
            return {
                "model_name": "None (Skipped)",
                "latency_ms": 0.0,
                "detections": []
            }

        if policy == "LIGHT":
            return self.lightweight.infer(frame_or_crop)

        if policy == "STANDARD":
            return self.standard.infer(frame_or_crop)

        if policy == "HIGH_ACCURACY":
            return self.high_accuracy.infer(frame_or_crop)

        return self.lightweight.infer(frame_or_crop)
