import cv2
import time
from typing import Dict, Any
from ultralytics import YOLO

from detection.event_detector import EventDetector
from adaptive.policy_engine import PolicyEngine
from evaluation.metrics import PerformanceMetrics

def run_system_a_always_on(video_path: str, model_path: str = "yolov8m.pt") -> Dict[str, Any]:
    metrics = PerformanceMetrics()
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, verbose=False)
        latency = (time.time() - start) * 1000.0

        detections = len(results[0].boxes) if results and results[0].boxes is not None else 0
        metrics.record_frame(inferences=1, latency_ms=latency, events=detections)

    cap.release()
    summary = metrics.get_summary()
    summary["system"] = "System A (Always-On)"
    return summary

def run_system_b_event_driven(video_path: str, config: Dict[str, Any], model_path: str = "yolov8m.pt") -> Dict[str, Any]:
    metrics = PerformanceMetrics()
    detector = EventDetector(config=config)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        events = detector.process_frame(frame)
        latency = (time.time() - start) * 1000.0

        inferences = 1 if len(events) > 0 else 0
        metrics.record_frame(inferences=inferences, latency_ms=latency, events=len(events))

    cap.release()
    summary = metrics.get_summary()
    summary["system"] = "System B (Event-Driven Static)"
    return summary

def run_system_c_eventvision(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    metrics = PerformanceMetrics()
    detector = EventDetector(config=config)
    policy_engine = PolicyEngine(config=config)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        start = time.time()
        raw_events = detector.process_frame(frame, frame_id=frame_id)

        inferences = 0
        cloud_synced = False

        for raw_evt in raw_events:
            processed_evt = policy_engine.process_event_adaptively(raw_evt, frame)
            if processed_evt.processing_level != "SKIP":
                inferences += 1
            if processed_evt.cloud_synced:
                cloud_synced = True

        latency = (time.time() - start) * 1000.0
        metrics.record_frame(inferences=inferences, latency_ms=latency, events=len(raw_events), cloud_synced=cloud_synced)

    cap.release()
    summary = metrics.get_summary()
    summary["system"] = "System C (EventVision Adaptive)"
    return summary
