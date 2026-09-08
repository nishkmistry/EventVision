from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.database import DatabaseManager
from events.event import Event

app = FastAPI(title="EventVision API", version="1.0.0", description="API for EventVision pipeline control, events, and performance metrics")

db = DatabaseManager()

class SystemHealth(BaseModel):
    status: str
    db_status: str

@app.get("/", response_model=Dict[str, str])
def root():
    return {"message": "EventVision Intelligent Event-Driven Adaptive Pipeline API"}

@app.get("/health", response_model=SystemHealth)
def health_check():
    return SystemHealth(status="ok", db_status="connected")

@app.get("/events", response_model=List[Dict[str, Any]])
def get_events(limit: int = 50):
    return db.get_events(limit=limit)

@app.post("/events", response_model=Dict[str, str])
def post_event(event: Event):
    db.save_event(event)
    return {"status": "success", "event_id": event.event_id}

@app.get("/metrics", response_model=List[Dict[str, Any]])
def get_metrics(limit: int = 50):
    return db.get_latest_metrics(limit=limit)
