import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

def is_point_in_polygon(point: Tuple[int, int], polygon: List[List[int]]) -> bool:
    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    res = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return res >= 0

class PriorityEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "confidence": 0.25,
            "severity": 0.25,
            "zone": 0.30,
            "urgency": 0.20
        }

    def calculate_priority(self, confidence: float, severity: float, zone_importance: float, urgency: float) -> float:
        w_c = self.weights.get("confidence", 0.25)
        w_s = self.weights.get("severity", 0.25)
        w_z = self.weights.get("zone", 0.30)
        w_u = self.weights.get("urgency", 0.20)

        score = w_c * confidence + w_s * severity + w_z * zone_importance + w_u * urgency
        return round(float(np.clip(score, 0.0, 1.0)), 4)

class EventClassifier:
    SEVERITY_MAP = {
        "MOTION": 0.2,
        "ANOMALY": 0.4,
        "PERSON_SMILING": 0.85,  # High severity for smiling person event
        "VEHICLE_DETECTED": 0.6,
        "PERSON_DETECTED": 0.7,
        "RESTRICTED_ZONE_INTRUSION": 0.95,
        "CRITICAL_ANOMALY": 1.0
    }

    SEMANTIC_WEIGHT_MAP = {
        "RESTRICTED_ZONE_INTRUSION": 1.00,
        "PERSON_SMILING": 0.90,
        "CRITICAL_ANOMALY": 0.95,
        "PERSON_DETECTED": 0.85,
        "VEHICLE_DETECTED": 0.75,
        "ANOMALY": 0.60,
        "MOTION": 0.30
    }

    def __init__(self, restricted_zones: Optional[List[Dict[str, Any]]] = None, priority_engine: Optional[PriorityEngine] = None):
        self.restricted_zones = restricted_zones or []
        self.priority_engine = priority_engine or PriorityEngine()

    def calculate_semantic_score(self, event_type: str, confidence: float, zone_importance: float) -> float:
        base_semantic = self.SEMANTIC_WEIGHT_MAP.get(event_type, 0.50)
        semantic_score = base_semantic * 0.6 + zone_importance * 0.2 + confidence * 0.2
        return round(float(np.clip(semantic_score, 0.0, 1.0)), 4)

    def classify_detections(self, detections: List[Dict[str, Any]], has_motion: bool, motion_ratio: float) -> List[Dict[str, Any]]:
        classified_events = []

        if not detections and has_motion:
            event_type = "MOTION"
            confidence = round(min(1.0, motion_ratio * 5.0), 2)
            severity = self.SEVERITY_MAP["MOTION"]
            zone_name = "NORMAL"
            zone_importance = 0.1
            urgency = 0.2
            priority = self.priority_engine.calculate_priority(confidence, severity, zone_importance, urgency)
            semantic = self.calculate_semantic_score(event_type, confidence, zone_importance)
            reason = f"Significant frame motion detected (ratio: {motion_ratio:.3f})"

            classified_events.append({
                "event_type": event_type,
                "confidence": confidence,
                "severity": severity,
                "zone": zone_name,
                "zone_importance": zone_importance,
                "urgency": urgency,
                "priority_score": priority,
                "semantic_score": semantic,
                "capture_reason": reason,
                "bbox": None
            })
            return classified_events

        for det in detections:
            label = det.get("label", "").lower()
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("confidence", 0.5)

            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            in_restricted = False
            zone_name = "NORMAL"
            zone_importance = 0.1

            for rz in self.restricted_zones:
                poly = rz.get("polygon", [])
                if is_point_in_polygon((center_x, center_y), poly):
                    in_restricted = True
                    zone_name = rz.get("name", "RESTRICTED_ZONE")
                    zone_importance = rz.get("importance", 1.0)
                    break

            if label == "person_smiling":
                event_type = "PERSON_SMILING"
                severity = self.SEVERITY_MAP["PERSON_SMILING"]
                urgency = 0.85
                reason = f"Person smiling detected with confidence {conf*100:.1f}%"
            elif label == "person":
                if in_restricted:
                    event_type = "RESTRICTED_ZONE_INTRUSION"
                    severity = self.SEVERITY_MAP["RESTRICTED_ZONE_INTRUSION"]
                    urgency = 0.95
                    reason = f"Person detected inside restricted zone ({zone_name})"
                else:
                    event_type = "PERSON_DETECTED"
                    severity = self.SEVERITY_MAP["PERSON_DETECTED"]
                    urgency = 0.6
                    reason = f"Person detected in monitored area with confidence {conf*100:.1f}%"
            elif label in ["car", "truck", "bus", "motorcycle", "vehicle"]:
                if in_restricted:
                    event_type = "RESTRICTED_ZONE_INTRUSION"
                    severity = self.SEVERITY_MAP["RESTRICTED_ZONE_INTRUSION"]
                    urgency = 0.9
                    reason = f"Vehicle detected inside restricted zone ({zone_name})"
                else:
                    event_type = "VEHICLE_DETECTED"
                    severity = self.SEVERITY_MAP["VEHICLE_DETECTED"]
                    urgency = 0.5
                    reason = f"Vehicle detected with confidence {conf*100:.1f}%"
            else:
                event_type = f"{label.upper()}_DETECTED"
                severity = self.SEVERITY_MAP.get(event_type, 0.4)
                urgency = 0.3
                reason = f"{label.title()} detected with confidence {conf*100:.1f}%"

            priority = self.priority_engine.calculate_priority(conf, severity, zone_importance, urgency)
            semantic = self.calculate_semantic_score(event_type, conf, zone_importance)

            classified_events.append({
                "event_type": event_type,
                "confidence": conf,
                "severity": severity,
                "zone": zone_name,
                "zone_importance": zone_importance,
                "urgency": urgency,
                "priority_score": priority,
                "semantic_score": semantic,
                "capture_reason": reason,
                "bbox": bbox
            })

        return classified_events
