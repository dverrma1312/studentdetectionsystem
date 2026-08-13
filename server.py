import cv2
import time
import json
import asyncio
import threading
import os
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from analytics_engine import CampusAnalyticsEngine
from generate_test_video import create_synthetic_stream

app = FastAPI(title="iCloudEMS Campus Intelligence API")

# Global State Variables
latest_frame = None
latest_telemetry = {}
active_connections: Set[WebSocket] = set()

# Thread lock & events
frame_lock = threading.Lock()
pipeline_thread = None
stop_pipeline_event = threading.Event()

# Configuration
current_source = "synthetic" # "synthetic", "webcam", "file"
uploaded_file_path = None
active_model = "yolov8s.pt"
active_conf = 0.10
active_resolution = 640
active_line_pos = 0.15

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)

def ai_pipeline_worker():
    """
    Decoupled background worker running the analytics pipeline asynchronously.
    Reads frames, processes them, and stores results in shared global state.
    """
    global latest_frame, latest_telemetry, current_source, uploaded_file_path
    
    print(f"🎬 Background AI Worker started. Source: '{current_source}', Model: '{active_model}'")
    
    # Initialize Analytics Engine
    engine = CampusAnalyticsEngine(
        model_name=active_model,
        conf_threshold=active_conf,
        line_position_ratio=active_line_pos,
        imgsz=active_resolution
    )
    
    # Open Video Source
    if current_source == "webcam":
        cap = cv2.VideoCapture(0)
    elif current_source == "file" and uploaded_file_path:
        cap = cv2.VideoCapture(uploaded_file_path)
    else:
        # Default Synthetic stream
        synthetic_path = "sample_stream.mp4"
        if not os.path.exists(synthetic_path):
            create_synthetic_stream(synthetic_path)
        cap = cv2.VideoCapture(synthetic_path)
        
    if not cap.isOpened():
        print("❌ Error: Could not open video stream source.")
        return

    # Frame-rate pacing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 60:
        fps = 30.0
    frame_delay = 1.0 / fps

    while not stop_pipeline_event.is_set():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            if current_source != "webcam":
                # Loop video files frame-by-frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
                
        # Process frame
        annotated_frame, telemetry = engine.process_frame(frame)
        
        # Write to global shared memory (thread-safe)
        with frame_lock:
            latest_frame = annotated_frame.copy()
            latest_telemetry = telemetry

        # Micro-delay frame rate pacing for non-live sources
        elapsed = time.time() - start_time
        sleep_time = max(0.001, frame_delay - elapsed)
        time.sleep(sleep_time)

    cap.release()
    print("🛑 Background AI Worker stopped.")


def start_worker():
    global pipeline_thread, stop_pipeline_event
    stop_pipeline_event.clear()
    pipeline_thread = threading.Thread(target=ai_pipeline_worker, daemon=True)
    pipeline_thread.start()


def stop_worker():
    global stop_pipeline_event, pipeline_thread
    stop_pipeline_event.set()
    if pipeline_thread:
        pipeline_thread.join(timeout=2.0)


@app.on_event("startup")
async def startup_event():
    # Start the worker thread on server boot
    start_worker()


@app.on_event("shutdown")
async def shutdown_event():
    stop_worker()


# Video MJPEG streaming generator
def generate_mjpeg_stream():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.03)
                continue
            # Encode frame to JPEG
            ret, encoded_jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame_bytes = encoded_jpeg.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Streaming frame-rate throttle (~30 FPS)


@app.get("/video_feed")
async def video_feed():
    """
    Serves the live annotated stream via MJPEG (multipart/x-mixed-replace).
    Provides zero-lag video rendering natively in the browser.
    """
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# WebSocket Broadcast for real-time telemetry every 100ms
async def broadcast_telemetry_loop():
    global latest_telemetry
    while True:
        if active_connections:
            # Prepare payload
            payload = json.dumps(latest_telemetry)
            # Broadcast to all connected clients
            disconnected = []
            for ws in active_connections:
                try:
                    await ws.send_text(payload)
                except WebSocketDisconnect:
                    disconnected.append(ws)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                active_connections.remove(ws)
        await asyncio.sleep(0.1) # Telemetry broadcast interval (100ms)


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for real-time telemetry updates.
    """
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.post("/api/update_settings")
async def update_settings(
    model: str = Form("yolov8s.pt"),
    conf: float = Form(0.10),
    resolution: int = Form(640),
    line_pos: float = Form(0.15),
    source: str = Form("synthetic")
):
    global active_model, active_conf, active_resolution, active_line_pos, current_source
    
    print(f"⚙️ Updating settings: Source={source}, Model={model}, Conf={conf}, Res={resolution}, Line={line_pos}")
    
    stop_worker()
    
    active_model = model
    active_conf = conf
    active_resolution = resolution
    active_line_pos = line_pos
    current_source = source
    
    start_worker()
    
    return {"status": "success", "message": "Pipeline settings updated and restarted."}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global uploaded_file_path, current_source
    
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"📁 Video file uploaded successfully: '{file_path}'")
    
    stop_worker()
    uploaded_file_path = file_path
    current_source = "file"
    start_worker()
    
    return {"status": "success", "file_path": file_path}


@app.get("/api/config")
async def get_config():
    return {
        "source": current_source,
        "model": active_model,
        "conf": active_conf,
        "resolution": active_resolution,
        "line_pos": active_line_pos
    }

# Start telemetry loop in background
@app.on_event("startup")
async def start_telemetry_broadcast():
    asyncio.create_task(broadcast_telemetry_loop())


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    # Return HTML index page directly
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
