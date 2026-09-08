import os
import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Any
from detection.motion import MotionDetector
from detection.object_detector import ObjectDetector
from events.classifier import EventClassifier, PriorityEngine
from events.event import Event

class EventDetector:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 object_detector: Optional[ObjectDetector] = None,
                 snapshots_dir: str = "data/snapshots"):
        self.config = config or {}
        det_cfg = self.config.get("detection", {})
        self.motion_detector = MotionDetector(
            threshold=det_cfg.get("motion_threshold", 25),
            min_contour_area=det_cfg.get("min_contour_area", 500)
        )
        self.object_detector = object_detector or ObjectDetector(
            model_name=det_cfg.get("yolo_model_default", "yolov8n.pt")
        )
        restricted_zones = self.config.get("restricted_zones", [])
        p_weights = self.config.get("priority", {}).get("weights", None)
        priority_engine = PriorityEngine(weights=p_weights)
        self.classifier = EventClassifier(restricted_zones=restricted_zones, priority_engine=priority_engine)
        self.snapshots_dir = snapshots_dir
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[Event]:
        if frame is None or frame.size == 0:
            return []

        start_time = time.time()
        has_motion, motion_bboxes, motion_ratio = self.motion_detector.detect(frame)

        if not has_motion:
            return []

        detections = self.object_detector.detect(frame, confidence_threshold=0.3)
        classified = self.classifier.classify_detections(detections, has_motion, motion_ratio)

        events = []
        latency_ms = (time.time() - start_time) * 1000.0

        for item in classified:
            snapshot_filename = f"event_{frame_id}_{int(time.time()*1000)}.jpg"
            snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)

            annotated_frame = frame.copy()
            bbox = item.get("bbox")
            if bbox:
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{item['event_type']} ({item['priority_score']})",
                            (bbox[0], max(20, bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imwrite(snapshot_path, annotated_frame)

            evt = Event(
                event_type=item["event_type"],
                confidence=item["confidence"],
                severity=item["severity"],
                zone=item["zone"],
                zone_importance=item["zone_importance"],
                urgency=item["urgency"],
                priority_score=item["priority_score"],
                bbox=bbox,
                image_snapshot_path=snapshot_path,
                latency_ms=round(latency_ms, 2)
            )
            events.append(evt)

        return events
