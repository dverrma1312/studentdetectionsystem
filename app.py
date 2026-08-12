import streamlit as st
import cv2
import tempfile
import time
import pandas as pd
from PIL import Image
from analytics_engine import CampusAnalyticsEngine
from generate_test_video import create_synthetic_stream

# Page configuration
st.set_page_config(
    page_title="iCloudEMS Campus Intelligence",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .alert-box {
        background-color: #3B1219;
        border: 1px solid #FF5252;
        border-radius: 8px;
        padding: 10px;
        color: #FF8A80;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 iCloudEMS Campus Intelligence Pipeline")
st.caption("Real-Time CCTV Live Stream Analytics — Attendance, Motion, Posture & Energy Optimization")

# Sidebar Configuration
st.sidebar.header("⚙️ Stream Controls & Settings")

input_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Synthetic Live Stream (Default)", "Webcam (Live)", "Upload Video File"],
    index=0
)

st.sidebar.subheader("🎯 Detection & Accuracy Controls")

model_choice = st.sidebar.selectbox(
    "YOLO Model Preset",
    options=[
        "yolov8s.pt (Recommended — Fast + Accurate)",
        "yolov8n.pt (Ultra Fast)",
        "yolov8m.pt (Maximum Accuracy — Slower)"
    ],
    index=0
)
model_name = model_choice.split()[0]

resolution_choice = st.sidebar.selectbox(
    "Scanning Resolution",
    options=[
        "640 (Standard — Fast & Smooth)",
        "1024 (High-Res — More Detail)",
        "1280 (Ultra-Res — Slower)"
    ],
    index=0
)
imgsz = int(resolution_choice.split()[0])

conf_thresh = st.sidebar.slider(
    "Detection Confidence Threshold", 0.05, 0.80, 0.08, 0.01,
    help="Lower = catches more people (including partially hidden). "
         "Try 0.08–0.12 for crowded classrooms."
)

line_pos = st.sidebar.slider(
    "Entry/Exit Line Position (from left edge)", 0.05, 0.50, 0.15, 0.05,
    help="Vertical line on the LEFT side simulating a doorway."
)

frame_sample_rate = st.sidebar.slider("Frame Sampling (Process 1 in N frames)", 1, 5, 2,
                                      help="2 = smooth playback, 1 = maximum accuracy but slower")

# File uploader
uploaded_file = None
if input_mode == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])

run_pipeline = st.sidebar.button("▶️ Start / Restart Live Stream", type="primary")

# Initialize Engine
engine = CampusAnalyticsEngine(
    model_name=model_name,
    conf_threshold=conf_thresh,
    line_position_ratio=line_pos,
    imgsz=imgsz
)

# UI Layout
col_video, col_telemetry = st.columns([2.2, 1.2])

with col_video:
    st.subheader("📹 Live Video Stream")
    video_placeholder = st.empty()

with col_telemetry:
    st.subheader("📊 Live Telemetry & Metrics")
    
    m1, m2 = st.columns(2)
    with m1:
        present_metric = st.empty()
    with m2:
        entries_metric = st.empty()
        
    m3, m4 = st.columns(2)
    with m3:
        posture_metric = st.empty()
    with m4:
        quality_metric = st.empty()
        
    m5, m6 = st.columns(2)
    with m5:
        occupancy_metric = st.empty()
    with m6:
        motion_metric = st.empty()

    utility_alert_placeholder = st.empty()

st.divider()
st.subheader("📜 Real-Time Activity Log")
log_placeholder = st.empty()

# Stream Processing
if run_pipeline or "streaming" in st.session_state:
    st.session_state["streaming"] = True
    
    if input_mode == "Webcam (Live)":
        cap = cv2.VideoCapture(0)
    elif input_mode == "Upload Video File" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        sample_video_path = "sample_stream.mp4"
        create_synthetic_stream(sample_video_path)
        cap = cv2.VideoCapture(sample_video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video source!")
    else:
        frame_count = 0
        
        while cap.isOpened() and st.session_state.get("streaming", False):
            ret, frame = cap.read()
            if not ret:
                if input_mode != "Webcam (Live)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            frame_count += 1
            if frame_count % frame_sample_rate != 0:
                continue

            annotated_frame, telemetry = engine.process_frame(frame)
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(Image.fromarray(frame_rgb), use_container_width=True)

            present_metric.metric("Currently Present", f"{telemetry['currently_present']} people")
            entries_metric.metric("Total Unique Entries", f"{telemetry['total_unique_entries']} people")
            posture_metric.metric("Seated vs Standing",
                                  f"🪑 {telemetry['seated_count']} | 🚶 {telemetry['standing_count']}")
            quality_metric.metric("Frame Quality", telemetry['blur_status'])
            occupancy_metric.metric("Room Occupancy", telemetry['occupancy_status'])
            motion_metric.metric("Motion", telemetry['motion_status'])

            if telemetry['wasted_utility_alert']:
                utility_alert_placeholder.markdown(
                    f"<div class='alert-box'>⚠️ UTILITY WARNING: {telemetry['utility_message']}</div>",
                    unsafe_allow_html=True
                )
            else:
                utility_alert_placeholder.success("✅ Utility Usage: Normal (No energy waste)")

            if telemetry['event_log']:
                log_df = pd.DataFrame(telemetry['event_log'][:15], columns=["Timestamped Events"])
                log_placeholder.dataframe(log_df, use_container_width=True, height=250)

            time.sleep(0.01)

        cap.release()
