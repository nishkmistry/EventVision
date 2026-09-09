import pytest
import numpy as np
import os
from detection.motion import MotionDetector
from detection.smile_detector import SmileDetector
from events.classifier import EventClassifier, PriorityEngine
from events.event import Event
from adaptive.policy_engine import PolicyEngine
from adaptive.resolution import adapt_resolution
from backend.database import DatabaseManager

@pytest.fixture
def sample_frame():
    frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
    frame[300:400, 500:600] = 200
    return frame

def test_motion_detector(sample_frame):
    md = MotionDetector(threshold=25, min_contour_area=100)
    md.detect(sample_frame)
    changed_frame = sample_frame.copy()
    changed_frame[100:200, 100:200] = 255
    has_motion, bboxes, ratio = md.detect(changed_frame)
    assert has_motion is True
    assert len(bboxes) > 0

def test_smile_detector_class():
    sd = SmileDetector()
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)
    res = sd.detect_smiles(dummy_frame)
    assert isinstance(res, list)

def test_priority_engine():
    pe = PriorityEngine()
    p_high = pe.calculate_priority(confidence=0.9, severity=0.95, zone_importance=1.0, urgency=0.9)
    assert p_high >= 0.85
    p_low = pe.calculate_priority(confidence=0.3, severity=0.2, zone_importance=0.1, urgency=0.2)
    assert p_low <= 0.35

def test_classifier():
    restricted_zones = [{
        "name": "Test_Zone",
        "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
        "importance": 1.0
    }]
    classifier = EventClassifier(restricted_zones=restricted_zones)
    detections = [{
        "label": "person",
        "confidence": 0.9,
        "bbox": [150, 150, 250, 250]
    }]
    events = classifier.classify_detections(detections, has_motion=True, motion_ratio=0.1)
    assert len(events) == 1
    assert events[0]["event_type"] == "RESTRICTED_ZONE_INTRUSION"
    assert events[0]["priority_score"] > 0.8
    assert "capture_reason" in events[0]

def test_adapt_resolution(sample_frame):
    resized, scale = adapt_resolution(sample_frame, level="LIGHT")
    assert resized.shape[0] == 360
    assert resized.shape[1] == 640

    bbox = [200, 200, 400, 400]
    cropped, _ = adapt_resolution(sample_frame, bbox=bbox, level="HIGH_ACCURACY")
    assert cropped.shape[0] < sample_frame.shape[0]

def test_policy_engine():
    config = {
        "priority": {"thresholds": {"skip": 0.2, "light": 0.5, "standard": 0.8, "high_accuracy": 1.0}}
    }
    pe = PolicyEngine(config=config)
    assert pe.determine_policy(0.1) == "SKIP"
    assert pe.determine_policy(0.4) == "LIGHT"
    assert pe.determine_policy(0.7) == "STANDARD"
    assert pe.determine_policy(0.9) == "HIGH_ACCURACY"

def test_database_manager(tmp_path):
    db_file = os.path.join(tmp_path, "test_events.db")
    db = DatabaseManager(db_path=db_file)
    evt = Event(
        event_type="PERSON_SMILING",
        confidence=0.92,
        severity=0.85,
        zone="Restricted_Zone_A",
        priority_score=0.88,
        capture_reason="Person smiling detected",
        processing_level="STANDARD",
        model_used="SmileCascade"
    )
    db.save_event(evt)
    retrieved = db.get_events(limit=1)
    assert len(retrieved) == 1
    assert retrieved[0]["event_id"] == evt.event_id
    assert retrieved[0]["capture_reason"] == "Person smiling detected"
