import os
import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Any
from detection.motion import MotionDetector
from detection.object_detector import ObjectDetector
from detection.smile_detector import SmileDetector
from events.classifier import EventClassifier, PriorityEngine
from events.event import Event

class EventDetector:
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 object_detector: Optional[ObjectDetector] = None,
                 snapshots_dir: str = "data/snapshots"):
        self.config = config or {}
        det_cfg = self.config.get("detection", {})
        p_cfg = self.config.get("priority", {})

        self.motion_detector = MotionDetector(
            threshold=det_cfg.get("motion_threshold", 25),
            min_contour_area=det_cfg.get("min_contour_area", 500)
        )
        self.object_detector = object_detector or ObjectDetector(
            model_name=det_cfg.get("yolo_model_default", "yolov8n.pt")
        )
        self.smile_detector = SmileDetector()

        # Trivial flicker/noise (tiny motion_ratio) never reaches YOLO at all.
        self.min_motion_ratio = det_cfg.get("min_motion_ratio", 0.015)

        restricted_zones = self.config.get("restricted_zones", [])
        p_weights = p_cfg.get("weights", None)
        priority_engine = PriorityEngine(weights=p_weights)
        self.classifier = EventClassifier(restricted_zones=restricted_zones, priority_engine=priority_engine)
        self.snapshots_dir = snapshots_dir
        self.snapshot_priority_threshold = p_cfg.get("snapshot_priority_threshold", 0.50)
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> List[Event]:
        if frame is None or frame.size == 0:
            return []

        start_time = time.time()
        has_motion, motion_bboxes, motion_ratio = self.motion_detector.detect(frame)

        # Detect smiles (always active on frame or motion)
        smile_detections = self.smile_detector.detect_smiles(frame) if has_motion else []

        # Ignore trivial background flicker before it ever reaches YOLO —
        # this alone eliminates most unnecessary inferences on a live feed.
        if not has_motion or (motion_ratio < self.min_motion_ratio and not smile_detections):
            return []

        # Motion / smile triggered automated object detection
        detections = self.object_detector.detect(frame, confidence_threshold=0.45)
        # Combine object detections with smile detections
        detections.extend(smile_detections)

        classified = self.classifier.classify_detections(detections, has_motion, motion_ratio)

        events = []
        latency_ms = (time.time() - start_time) * 1000.0

        for item in classified:
            priority_score = item["priority_score"]
            snapshot_path = None

            # Capture snapshot image during significant activity or when person smiles
            if priority_score >= self.snapshot_priority_threshold or item["event_type"] == "PERSON_SMILING":
                snapshot_filename = f"event_{frame_id}_{int(time.time()*1000)}.jpg"
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_filename)

                annotated_frame = frame.copy()
                bbox = item.get("bbox")
                if bbox:
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    label = f"{item['event_type']} (P:{priority_score:.2f})"
                    cv2.putText(annotated_frame, label, (bbox[0], max(20, bbox[1] - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Annotate image snapshot with clear description of WHY image was captured
                reason_text = f"REASON: {item['capture_reason']}"
                cv2.rectangle(annotated_frame, (10, 10), (frame.shape[1] - 10, 50), (0, 0, 0), -1)
                cv2.putText(annotated_frame, reason_text, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                cv2.imwrite(snapshot_path, annotated_frame)

            evt = Event(
                event_type=item["event_type"],
                confidence=item["confidence"],
                severity=item["severity"],
                zone=item["zone"],
                zone_importance=item["zone_importance"],
                urgency=item["urgency"],
                priority_score=priority_score,
                semantic_score=item.get("semantic_score", 0.5),
                capture_reason=item.get("capture_reason", "Detected activity"),
                bbox=item.get("bbox"),
                image_snapshot_path=snapshot_path,
                latency_ms=round(latency_ms, 2)
            )
            events.append(evt)

        return events
