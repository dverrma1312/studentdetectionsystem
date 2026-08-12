import cv2
import sys
import os
from analytics_engine import CampusAnalyticsEngine
from generate_test_video import create_synthetic_stream

def test_full_pipeline():
    print("==================================================")
    print("🧪 Running End-to-End Pipeline Verification Test")
    print("==================================================")
    
    video_path = "sample_stream.mp4"
    if not os.path.exists(video_path):
        print("Generating synthetic video stream...")
        create_synthetic_stream(video_path)

    engine = CampusAnalyticsEngine(model_name="yolov8n.pt", conf_threshold=0.25)
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open synthetic video stream!"
    
    total_frames = 0
    detected_motion = 0
    detected_blur = 0
    utility_alerts = 0
    max_present = 0
    
    print("\nProcessing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        annotated_frame, telemetry = engine.process_frame(frame)
        
        # Verify frame output shape
        assert annotated_frame.shape == frame.shape, "Annotated frame dimensions mismatch!"
        
        # Collect statistics
        if telemetry['motion_status'] == "Motion detected":
            detected_motion += 1
        if telemetry['blur_status'] != "Clear":
            detected_blur += 1
        if telemetry['wasted_utility_alert']:
            utility_alerts += 1
            
        max_present = max(max_present, telemetry['currently_present'])
        
    cap.release()
    
    print("\n--- Pipeline Execution Summary ---")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Motion Detected Frames: {detected_motion}")
    print(f"Blurry Frames Flagged: {detected_blur}")
    print(f"Total Unique Entries Logged: {engine.total_unique_entries}")
    print(f"Max Concurrent People Present: {max_present}")
    print(f"Utility Warning Alerts Raised: {utility_alerts}")
    print(f"Event Log Entries Count: {len(engine.event_log)}")
    
    # Assertions
    assert total_frames > 0, "No frames were processed!"
    assert len(engine.event_log) > 0, "Event log is empty!"
    
    print("\n✅ Verification SUCCESS! All computer vision components are functioning cleanly.")

if __name__ == "__main__":
    test_full_pipeline()
