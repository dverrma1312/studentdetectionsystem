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

# Custom CSS for dark theme aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2E3440;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00E676;
    }
    .metric-label {
        font-size: 14px;
        color: #A0AAB0;
    }
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
    options=["yolov8s.pt (Recommended - High Classroom Accuracy)", "yolov8m.pt (Maximum Classroom Accuracy)", "yolov8n.pt (Ultra Fast)"],
    index=0
)
model_name = model_choice.split()[0]

resolution_choice = st.sidebar.selectbox(
    "Scanning Resolution (detects small/far people)",
    options=["1024 (High-Res - Full Classroom Coverage)", "1280 (Ultra-Res - Maximum Detail)", "640 (Standard)"],
    index=0
)
imgsz = int(resolution_choice.split()[0])

conf_thresh = st.sidebar.slider(
    "Detection Confidence Threshold", 0.05, 0.80, 0.15, 0.05, 
    help="Lower threshold (0.10 - 0.20) detects students sitting in the back rows and left desks."
)

line_pos = st.sidebar.slider("Entry/Exit Boundary Line Position", 0.10, 0.90, 0.50, 0.05)
frame_sample_rate = st.sidebar.slider("Frame Sampling (Process 1 in N frames)", 1, 5, 1)

# Ensure sample video exists
sample_video_path = "sample_stream.mp4"

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

# UI Layout (Video Player on left, Telemetry on right)
col_video, col_telemetry = st.columns([2.2, 1.2])

with col_video:
    st.subheader("📹 Live Video Stream")
    video_placeholder = st.empty()

with col_telemetry:
    st.subheader("📊 Live Telemetry & Metrics")
    
    # Telemetry placeholders
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        present_metric = st.empty()
    with m_col2:
        entries_metric = st.empty()
        
    m_col3, m_col4 = st.columns(2)
    with m_col3:
        posture_metric = st.empty()
    with m_col4:
        quality_metric = st.empty()
        
    m_col5, m_col6 = st.columns(2)
    with m_col5:
        occupancy_metric = st.empty()
    with m_col6:
        motion_metric = st.empty()

    utility_alert_placeholder = st.empty()

st.divider()
st.subheader("📜 Real-Time Activity Log")
log_placeholder = st.empty()

# Stream Processing Execution
if run_pipeline or "streaming" in st.session_state:
    st.session_state["streaming"] = True
    
    # Determine video capture source
    if input_mode == "Webcam (Live)":
        cap = cv2.VideoCapture(0)
    elif input_mode == "Upload Video File" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        # Default Synthetic Stream
        create_synthetic_stream(sample_video_path)
        cap = cv2.VideoCapture(sample_video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video source!")
    else:
        frame_count = 0
        
        while cap.isOpened() and st.session_state.get("streaming", False):
            ret, frame = cap.read()
            if not ret:
                # Loop video if source is file
                if input_mode != "Webcam (Live)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            frame_count += 1
            if frame_count % frame_sample_rate != 0:
                continue

            # Process Frame through Analytics Engine
            annotated_frame, telemetry = engine.process_frame(frame)
            
            # Convert OpenCV frame (BGR) to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            video_placeholder.image(img, use_container_width=True)

            # Update Telemetry Metrics
            present_metric.metric("Currently Present", f"{telemetry['currently_present']} people")
            entries_metric.metric("Total Unique Entries", f"{telemetry['total_unique_entries']} people")
            posture_metric.metric("Seated vs Standing", f"🪑 {telemetry['seated_count']} | 🚶 {telemetry['standing_count']}")
            
            # Status badges
            quality_metric.metric("Frame Quality", telemetry['blur_status'])
            occupancy_metric.metric("Room Occupancy", telemetry['occupancy_status'])
            motion_metric.metric("Motion", telemetry['motion_status'])

            # Wasted Utility Banner
            if telemetry['wasted_utility_alert']:
                utility_alert_placeholder.markdown(
                    f"<div class='alert-box'>⚠️ UTILITY WARNING: {telemetry['utility_message']}</div>",
                    unsafe_allow_html=True
                )
            else:
                utility_alert_placeholder.success("✅ Utility Usage: Normal (No energy waste)")

            # Event Log Table
            if telemetry['event_log']:
                log_df = pd.DataFrame(telemetry['event_log'][:10], columns=["Timestamped Events"])
                log_placeholder.dataframe(log_df, use_container_width=True, height=200)

            time.sleep(0.01)

        cap.release()
