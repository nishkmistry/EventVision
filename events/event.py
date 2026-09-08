import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    confidence: float
    severity: float
    zone: str = "DEFAULT"
    zone_importance: float = 0.0
    urgency: float = 0.5
    priority_score: float = 0.0
    semantic_score: float = 0.5  # Contextual semantic relevance score [0.0 to 1.0]
    processing_level: str = "SKIP"
    model_used: str = "None"
    latency_ms: float = 0.0
    bbox: Optional[List[int]] = None
    image_snapshot_path: Optional[str] = None
    cloud_synced: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
