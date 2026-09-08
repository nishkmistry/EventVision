# EventVision: Intelligent Event-Driven Adaptive Image Processing Pipeline with Dynamic AI Model Selection

EventVision is an intelligent, event-driven image processing and computer vision pipeline designed to drastically reduce compute, latency, and communication overhead compared to always-on inference systems.

## Key Features
- **YOLO-Powered Automated Event Detection**: Automatically detects motion, people, vehicles, and restricted zone intrusions.
- **Priority Engine & Zone Detection**: Computes event priority scores $P = w_1 C + w_2 S + w_3 Z + w_4 U$.
- **Adaptive Policy & Dynamic Model Selection**: Dynamically routes events between Lightweight (`YOLOv8n`), Standard (`YOLOv8s`), and High Accuracy (`YOLOv8m`) models.
- **Adaptive Resolution & Crop**: Downscales or crops Regions of Interest (ROI) based on priority.
- **AWS Cloud Router**: Automatically syncs high-priority event snapshots to AWS S3 / Cloud infrastructure.
- **Streamlit Dashboard & FastAPI**: Real-time live video visualization and REST API endpoints.
- **Benchmarking Suite**: Evaluates System A (Always-On) vs. System B (Static Event) vs. System C (EventVision Adaptive).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Sample Surveillance Video
```bash
python3 data/generate_sample_video.py
```

### 3. Run Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Run API Server
```bash
uvicorn backend.api:app --reload
```

### 5. Run Benchmark Comparison
```bash
PYTHONPATH=. python3 evaluation/benchmark.py
```

### 6. Run Tests
```bash
PYTHONPATH=. pytest tests/
```
