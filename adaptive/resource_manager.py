import psutil
import time
from typing import Dict, Any

class ResourceManager:
    def __init__(self, cpu_threshold: float = 85.0, ram_threshold: float = 85.0):
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold

    def get_system_metrics(self) -> Dict[str, Any]:
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent
        return {
            "timestamp": time.time(),
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "gpu_usage": 0.0,
            "is_throttled": cpu_usage > self.cpu_threshold or ram_usage > self.ram_threshold
        }
