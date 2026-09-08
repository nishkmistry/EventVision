import time
from typing import Dict, Any

class PerformanceMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.ai_inferences = 0
        self.total_latency_ms = 0.0
        self.events_detected = 0
        self.cloud_syncs = 0

    def record_frame(self, inferences: int, latency_ms: float, events: int = 0, cloud_synced: bool = False):
        self.total_frames += 1
        self.ai_inferences += inferences
        self.total_latency_ms += latency_ms
        self.events_detected += events
        if cloud_synced:
            self.cloud_syncs += 1

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = (self.total_latency_ms / self.total_frames) if self.total_frames > 0 else 0.0
        return {
            "total_frames": self.total_frames,
            "ai_inferences": self.ai_inferences,
            "avg_latency_ms": round(avg_latency, 2),
            "events_detected": self.events_detected,
            "cloud_syncs": self.cloud_syncs
        }
