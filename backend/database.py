import sqlite3
import json
from typing import List, Dict, Any
from events.event import Event

class DatabaseManager:
    def __init__(self, db_path: str = "eventvision.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                confidence REAL,
                severity REAL,
                zone TEXT,
                zone_importance REAL,
                urgency REAL,
                priority_score REAL,
                semantic_score REAL,
                processing_level TEXT,
                model_used TEXT,
                latency_ms REAL,
                bbox TEXT,
                image_snapshot_path TEXT,
                cloud_synced INTEGER,
                metadata TEXT
            )
            """)
            # Check if semantic_score column exists (for schema migrations if table existed)
            cursor.execute("PRAGMA table_info(events)")
            columns = [col[1] for col in cursor.fetchall()]
            if "semantic_score" not in columns:
                cursor.execute("ALTER TABLE events ADD COLUMN semantic_score REAL DEFAULT 0.5")

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_usage REAL,
                ram_usage REAL,
                gpu_usage REAL,
                fps REAL,
                events_count INTEGER
            )
            """)
            conn.commit()

    def save_event(self, event: Event):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO events (
                event_id, timestamp, event_type, confidence, severity, zone,
                zone_importance, urgency, priority_score, semantic_score, processing_level,
                model_used, latency_ms, bbox, image_snapshot_path, cloud_synced, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.timestamp,
                event.event_type,
                event.confidence,
                event.severity,
                event.zone,
                event.zone_importance,
                event.urgency,
                event.priority_score,
                event.semantic_score,
                event.processing_level,
                event.model_used,
                event.latency_ms,
                json.dumps(event.bbox) if event.bbox else None,
                event.image_snapshot_path,
                1 if event.cloud_synced else 0,
                json.dumps(event.metadata)
            ))
            conn.commit()

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            events = []
            for row in rows:
                d = dict(row)
                d["bbox"] = json.loads(d["bbox"]) if d["bbox"] else None
                d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
                d["cloud_synced"] = bool(d["cloud_synced"])
                events.append(d)
            return events

    def save_metrics(self, timestamp: float, cpu_usage: float, ram_usage: float, gpu_usage: float, fps: float, events_count: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO system_metrics (timestamp, cpu_usage, ram_usage, gpu_usage, fps, events_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, cpu_usage, ram_usage, gpu_usage, fps, events_count))
            conn.commit()

    def get_latest_metrics(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM system_metrics ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]
