import streamlit as st
import cv2
import pandas as pd
import yaml
import time
import os
import psutil

from detection.event_detector import EventDetector
from adaptive.policy_engine import PolicyEngine
from backend.database import DatabaseManager

st.set_page_config(
    page_title="EventVision Dashboard",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_config_and_components():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    detector = EventDetector(config=config)
    policy_engine = PolicyEngine(config=config)
    db = DatabaseManager(config.get("database", {}).get("db_path", "eventvision.db"))
    return config, detector, policy_engine, db

config, detector, policy_engine, db = load_config_and_components()

st.title("👁️ EventVision: Adaptive Image Processing Pipeline")
st.caption("Intelligent Real-Time Event Detection, Dynamic Model Selection & Semantic Context Engine")

st.sidebar.header("🕹️ Pipeline Controls")
input_source_raw = st.sidebar.text_input("Camera Index or Video File Path", value=str(config.get("video", {}).get("source", 0)))

if input_source_raw.isdigit():
    video_source = int(input_source_raw)
else:
    video_source = input_source_raw

frame_step = st.sidebar.slider("Frame Processing Sampling Step (higher = smoother / no lag)", min_value=1, max_value=10, value=config.get("video", {}).get("process_every_n_frames", 3))
snapshot_threshold = st.sidebar.slider("Significant Event Snapshot Priority Threshold", min_value=0.1, max_value=1.0, value=float(config.get("priority", {}).get("snapshot_priority_threshold", 0.50)), step=0.05)
detector.snapshot_priority_threshold = snapshot_threshold

aws_enabled = st.sidebar.checkbox("Enable AWS S3 Cloud Sync", value=config.get("aws", {}).get("enabled", False))

st.sidebar.divider()
st.sidebar.subheader("⚙️ System Metrics")
cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().percent
st.sidebar.progress(cpu_usage / 100.0, text=f"CPU Usage: {cpu_usage}%")
st.sidebar.progress(ram_usage / 100.0, text=f"RAM Usage: {ram_usage}%")

col_video, col_events = st.columns([3, 2])

with col_video:
    st.subheader("📹 Real-Time Stream & Event Overlay")
    video_placeholder = st.empty()
    status_placeholder = st.empty()

with col_events:
    st.subheader("⚡ Significant Activity & Semantic Context")
    event_list_placeholder = st.empty()

st.divider()
st.subheader("📊 System Metrics & Event Log")
tab1, tab2 = st.tabs(["Captured Significant Events", "Performance Metrics"])

if st.button("▶️ Start Live Stream / Camera"):
    if isinstance(video_source, str) and not os.path.exists(video_source):
        st.error(f"Video source path not found at: {video_source}")
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Could not open camera/video source: {video_source}")
        else:
            frame_idx = 0
            events_processed_count = 0
            start_time = time.time()
            stop_button = st.button("⏹️ Stop Stream")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame sampling step to eliminate video lag
                if frame_idx % frame_step != 0:
                    continue

                raw_events = detector.process_frame(frame, frame_id=frame_idx)
                display_frame = frame.copy()
                processed_events = []

                for raw_evt in raw_events:
                    evt = policy_engine.process_event_adaptively(raw_evt, frame)
                    db.save_event(evt)
                    processed_events.append(evt)
                    events_processed_count += 1

                    bbox = evt.bbox
                    if bbox:
                        color = (0, 0, 255) if evt.priority_score > 0.8 else (0, 255, 255) if evt.priority_score > 0.5 else (0, 255, 0)
                        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        label = f"{evt.event_type} | P:{evt.priority_score:.2f} S:{evt.semantic_score:.2f} | {evt.processing_level}"
                        cv2.putText(display_frame, label, (bbox[0], max(25, bbox[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                fps = frame_idx / (time.time() - start_time)
                status_placeholder.info(f"Frame: {frame_idx} | FPS: {fps:.1f} | Significant Events: {events_processed_count}")

                if processed_events:
                    latest_evt = processed_events[-1]
                    event_list_placeholder.markdown(f"""
                    **Latest Significant Event:**
                    - **Event Type:** `{latest_evt.event_type}`
                    - **Priority Score:** `{latest_evt.priority_score:.4f}`
                    - **Semantic Score:** `{latest_evt.semantic_score:.4f}`
                    - **Processing Policy:** `{latest_evt.processing_level}`
                    - **Model Used:** `{latest_evt.model_used}`
                    - **Snapshot Image:** `{latest_evt.image_snapshot_path or 'Not Triggered (below threshold)'}`
                    """)
                    if latest_evt.image_snapshot_path and os.path.exists(latest_evt.image_snapshot_path):
                        st.image(latest_evt.image_snapshot_path, caption=f"Snapshot: {latest_evt.event_type} (Priority: {latest_evt.priority_score})")

                time.sleep(0.001)

            cap.release()
            st.success("Stream stopped.")

with tab1:
    events_data = db.get_events(limit=50)
    if events_data:
        df_events = pd.DataFrame(events_data)
        st.dataframe(df_events[["timestamp", "event_type", "priority_score", "semantic_score", "processing_level", "model_used", "image_snapshot_path", "cloud_synced"]], use_container_width=True)
    else:
        st.info("No events captured yet.")

with tab2:
    metrics_data = db.get_latest_metrics(limit=50)
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        st.line_chart(df_metrics[["cpu_usage", "ram_usage", "fps"]])
    else:
        st.info("Run evaluation or processing to populate metrics.")
