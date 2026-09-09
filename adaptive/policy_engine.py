from typing import Dict, Any, Optional
from events.event import Event
from adaptive.resource_manager import ResourceManager
from adaptive.resolution import adapt_resolution
from adaptive.model_router import ModelRouter
from aws.cloud_adapter import AWSCloudAdapter

class PolicyEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        p_cfg = self.config.get("priority", {}).get("thresholds", {})
        self.thresh_skip = p_cfg.get("skip", 0.20)
        self.thresh_light = p_cfg.get("light", 0.50)
        self.thresh_standard = p_cfg.get("standard", 0.80)
        self.thresh_high = p_cfg.get("high_accuracy", 1.00)

        self.resource_manager = ResourceManager()
        self.model_router = ModelRouter(self.config.get("models"))
        self.aws_adapter = AWSCloudAdapter(self.config)

    def determine_policy(self, priority_score: float, is_throttled: bool = False) -> str:
        if priority_score < self.thresh_skip:
            return "SKIP"

        if is_throttled:
            if priority_score < self.thresh_standard:
                return "LIGHT"
            return "STANDARD"

        if priority_score < self.thresh_light:
            return "LIGHT"
        elif priority_score < self.thresh_standard:
            return "STANDARD"
        else:
            return "HIGH_ACCURACY"

    def process_event_adaptively(self, event: Event, frame) -> Event:
        metrics = self.resource_manager.get_system_metrics()
        policy = self.determine_policy(event.priority_score, metrics["is_throttled"])
        event.processing_level = policy

        if policy == "SKIP":
            event.model_used = "Skipped"
            return event

        if policy == "LIGHT":
            # EventDetector's initial classification pass already ran YOLO
            # (yolov8n) on this frame to produce the event. LIGHT policy
            # uses the same lightweight model, so re-running inference here
            # is pure duplicated cost with no accuracy benefit — reuse the
            # existing detection instead. STANDARD/HIGH_ACCURACY still get a
            # genuine second pass with a stronger model below.
            event.model_used = "yolov8n.pt (reused from detection stage)"
            return event

        adapted_frame, scale = adapt_resolution(frame, event.bbox, policy)
        result = self.model_router.route_and_process(policy, adapted_frame)
        event.model_used = result["model_name"]
        event.latency_ms += result["latency_ms"]

        if policy == "HIGH_ACCURACY" and event.image_snapshot_path:
            cloud_res = self.aws_adapter.upload_snapshot_to_s3(event.image_snapshot_path)
            event.cloud_synced = cloud_res.get("uploaded", False)
            event.metadata["aws_cloud_res"] = cloud_res

        return event
