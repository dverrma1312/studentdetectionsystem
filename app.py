import streamlit as st
import cv2
import tempfile
import time
import pandas as pd
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
from queue import Queue

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
    options=["Synthetic Live Stream (Default)", "Webcam (Live WebRTC)", "Upload Video File"],
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

# Global configuration dictionary passed to WebRTC processor
config = {
    "model_name": model_name,
    "conf_threshold": conf_thresh,
    "line_position_ratio": line_pos,
    "imgsz": imgsz
}

# WebRTC Video Processor class for real-time lag-free browser streaming
class WebRTCVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = CampusAnalyticsEngine(
            model_name=config["model_name"],
            conf_threshold=config["conf_threshold"],
            line_position_ratio=config["line_position_ratio"],
            imgsz=config["imgsz"]
        )
        self.telemetry = None
        self._lock = threading.Lock()

    def update_settings(self, model_name, conf, line, imgsz):
        with self._lock:
            # Recreate engine only if settings change to avoid drop frames
            if (self.engine.model.names[0] != model_name or 
                self.engine.conf_threshold != conf or 
                self.engine.line_position_ratio != line or
                self.engine.imgsz != imgsz):
                self.engine = CampusAnalyticsEngine(
                    model_name=model_name,
                    conf_threshold=conf,
                    line_position_ratio=line,
                    imgsz=imgsz
                )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Lock setting check
        with self._lock:
            annotated_frame, telemetry = self.engine.process_frame(img)
            self.telemetry = telemetry
            
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Initialize normal engine for offline file processing
@st.cache_resource
def get_file_engine(model, conf, line, resolution):
    return CampusAnalyticsEngine(
        model_name=model,
        conf_threshold=conf,
        line_position_ratio=line,
        imgsz=resolution
    )

engine = get_file_engine(model_name, conf_thresh, line_pos, imgsz)

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

# Webcam Stream processing using WebRTC (buttery smooth in browser)
if input_mode == "Webcam (Live WebRTC)":
    # Configure stun servers for WebRTC
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    ctx = webrtc_streamer(
        key="campus-intelligence",
        video_processor_factory=WebRTCVideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if ctx.video_processor:
        # Dynamically update settings inside WebRTC thread
        ctx.video_processor.update_settings(model_name, conf_thresh, line_pos, imgsz)
        
        # Display telemetry asynchronously
        telemetry = ctx.video_processor.telemetry
        if telemetry:
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

# Video File Stream Processing (Multi-threaded queue-based reader for zero-jitter file streaming)
else:
    uploaded_file = None
    if input_mode == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])
        
    run_pipeline = st.sidebar.button("▶️ Start / Restart Live Stream", type="primary")

    if run_pipeline or "streaming" in st.session_state:
        st.session_state["streaming"] = True
        
        if input_mode == "Upload Video File" and uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            video_source = tfile.name
        else:
            video_source = "sample_stream.mp4"
            create_synthetic_stream(video_source)

        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error("Error: Could not open video source!")
        else:
            # Multi-threaded Frame Producer-Consumer Queue to decouple disk read from UI thread
            frame_queue = Queue(maxsize=3)
            stop_thread = threading.Event()
            
            def frame_producer():
                while cap.isOpened() and not stop_thread.is_set():
                    ret, f = cap.read()
                    if not ret:
                        # Loop video files
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    try:
                        frame_queue.put(f, timeout=1.0)
                    except Exception:
                        pass
                cap.release()

            # Start background thread
            producer_thread = threading.Thread(target=frame_producer, daemon=True)
            producer_thread.start()
            
            while st.session_state.get("streaming", False):
                if frame_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                frame = frame_queue.get()
                
                # Process frame
                annotated_frame, telemetry = engine.process_frame(frame)
                
                # Render frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(Image.fromarray(frame_rgb), use_container_width=True)

                # Metrics update
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

                # Small sleep to smooth frame rates (approx 25 fps)
                time.sleep(0.03)

            stop_thread.set()
            producer_thread.join()
