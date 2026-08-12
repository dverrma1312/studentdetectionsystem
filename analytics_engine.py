import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

class CampusAnalyticsEngine:
    def __init__(
        self,
        model_name="yolov8s.pt",  # Upgraded to yolov8s for much higher classroom detection accuracy
        conf_threshold=0.20,       # Lowered default threshold from 0.35 to 0.20 for seated/occluded students
        blur_threshold=70.0,
        motion_threshold=0.015,
        light_threshold=140.0,
        line_position_ratio=0.5
    ):
        """
        Campus Intelligence Analytics Engine.
        Processes video frames real-time and computes room analytics.
        """
        # Load YOLO model
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # Thresholds
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        self.light_threshold = light_threshold
        self.line_position_ratio = line_position_ratio
        
        # State variables
        self.prev_gray = None
        self.total_unique_entries = 0
        self.active_tracks = {} # track_id -> {'last_y': float, 'state': 'outside'/'inside'}
        self.track_history = defaultdict(list) # track_id -> list of centroids (x,y)
        self.track_postures = {} # track_id -> 'Seated' / 'Standing/Moving'
        
        # Fallback centroid tracker state
        self.next_fallback_id = 1
        self.prev_centroids = {} # track_id -> (cx, cy)
        
        # Event Log
        self.event_log = []
        self.last_event = "System initialized"
        
    def _log_event(self, message):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self.last_event = message
        self.event_log.insert(0, formatted)
        if len(self.event_log) > 50:
            self.event_log.pop()

    def check_blur(self, frame):
        """
        Determines frame quality using Laplacian Variance.
        Low variance (< threshold) indicates a blurry frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < self.blur_threshold:
            status = "Blurry — flag for review"
        else:
            status = "Clear"
        return variance, status

    def detect_motion(self, frame):
        """
        Detects motion between consecutive frames using absdiff thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return "No motion", 0.0
            
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_ratio = np.sum(thresh == 255) / (frame.shape[0] * frame.shape[1])
        
        self.prev_gray = gray
        
        if motion_ratio > self.motion_threshold:
            return "Motion detected", motion_ratio
        return "No motion", motion_ratio

    def check_wasted_utility(self, frame, currently_present):
        """
        Checks if lights are left ON when the room is empty.
        Calculates average brightness in the V channel (HSV).
        """
        if currently_present > 0:
            return False, "Normal (Occupied)"
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        avg_brightness = np.mean(v_channel)
        
        if avg_brightness > self.light_threshold:
            return True, "Alert: Lights ON in Empty Room!"
        return False, "Normal (Empty & Off)"

    def _assign_fallback_ids(self, centroids, max_distance=70):
        """
        Centroid distance matching fallback if ByteTrack is unavailable.
        """
        assigned_ids = []
        new_prev_centroids = {}
        
        for cx, cy in centroids:
            best_id = None
            min_dist = float("inf")
            for tid, (px, py) in self.prev_centroids.items():
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_id = tid
                    
            if best_id is None:
                best_id = self.next_fallback_id
                self.next_fallback_id += 1
                
            assigned_ids.append(best_id)
            new_prev_centroids[best_id] = (cx, cy)
            
        self.prev_centroids = new_prev_centroids
        return assigned_ids

    def process_frame(self, frame):
        """
        Main pipeline function with improved classroom person detection & posture classification.
        """
        h, w = frame.shape[:2]
        crossing_y = int(h * self.line_position_ratio)
        
        # 1. Blur check
        blur_val, blur_status = self.check_blur(frame)
        
        # 2. Motion check
        motion_status, motion_ratio = self.detect_motion(frame)
        
        # 3. YOLO Tracking / Detection (Optimized for classroom crowd)
        annotated_frame = frame.copy()
        current_tracked_ids = set()
        seated_count = 0
        standing_count = 0
        
        try:
            results = self.model.track(
                source=frame,
                classes=[0],  # Person class
                conf=self.conf_threshold,
                persist=True,
                verbose=False
            )
        except Exception:
            results = self.model(
                source=frame,
                classes=[0],
                conf=self.conf_threshold,
                verbose=False
            )
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            
            if len(xyxy_boxes) > 0:
                centroids = [((int(b[0]) + int(b[2])) // 2, (int(b[1]) + int(b[3])) // 2) for b in xyxy_boxes]
                
                # Check if ByteTrack assigned IDs
                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = self._assign_fallback_ids(centroids)
                    
                for box, (center_x, center_y), track_id in zip(xyxy_boxes, centroids, track_ids):
                    current_tracked_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    box_w = max(x2 - x1, 1)
                    box_h = max(y2 - y1, 1)
                    
                    # Update track history
                    self.track_history[track_id].append((center_x, center_y))
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)
                        
                    # Calculate movement velocity over recent frames
                    history = self.track_history[track_id]
                    if len(history) >= 5:
                        dx = history[-1][0] - history[0][0]
                        dy = history[-1][1] - history[0][1]
                        velocity = np.sqrt(dx**2 + dy**2)
                    else:
                        velocity = 0.0
                        
                    # Enhanced Posture Classification Heuristic:
                    # Seated students stay mostly stationary (low velocity < 12.0)
                    # or have wider/squarish aspect ratio (height/width < 1.8).
                    aspect_ratio = box_h / box_w
                    if velocity < 12.0 or aspect_ratio < 1.4:
                        posture = "Seated"
                        seated_count += 1
                        box_color = (255, 191, 0) # Deep Cyan / Amber
                    else:
                        posture = "Standing/Moving"
                        standing_count += 1
                        box_color = (0, 255, 127) # Neon Green
                        
                    self.track_postures[track_id] = posture
                    
                    # Line Crossing Logic:
                    if track_id not in self.active_tracks:
                        initial_state = "inside" if center_y > crossing_y else "outside"
                        self.active_tracks[track_id] = {
                            'last_y': center_y,
                            'state': initial_state
                        }
                    else:
                        prev_state = self.active_tracks[track_id]['state']
                        if prev_state == "outside" and center_y >= crossing_y:
                            self.active_tracks[track_id]['state'] = "inside"
                            self.total_unique_entries += 1
                            self._log_event(f"Entry detected (Person #{track_id})")
                        elif prev_state == "inside" and center_y < crossing_y:
                            self.active_tracks[track_id]['state'] = "outside"
                            self._log_event(f"Exit detected (Person #{track_id})")
                            
                        self.active_tracks[track_id]['last_y'] = center_y
                    
                    # Draw Bounding Box & Label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    label = f"ID:#{track_id} | {posture}"
                    
                    (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - 22), (x1 + w_lbl, y1), box_color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    cv2.circle(annotated_frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # Count currently present inside room
        currently_present = sum(
            1 for tid, data in self.active_tracks.items()
            if tid in current_tracked_ids and data['state'] == "inside"
        )
        
        # Room Occupancy Status
        occupancy_status = "Occupied" if currently_present > 0 else "Empty"
        
        # 6. Wasted Utility Alert Check
        wasted_flag, utility_msg = self.check_wasted_utility(frame, currently_present)
        if wasted_flag and "Lights ON" not in self.last_event:
            self._log_event("UTILITY ALERT: Lights left ON in empty room!")
            
        # Draw Entry/Exit Virtual Line
        cv2.line(annotated_frame, (0, crossing_y), (w, crossing_y), (0, 0, 255), 2)
        cv2.putText(annotated_frame, "ENTRY / EXIT BOUNDARY LINE", (10, crossing_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Telemetry payload
        telemetry = {
            "currently_present": currently_present,
            "total_unique_entries": self.total_unique_entries,
            "seated_count": seated_count,
            "standing_count": standing_count,
            "blur_value": round(blur_val, 1),
            "blur_status": blur_status,
            "motion_status": motion_status,
            "occupancy_status": occupancy_status,
            "wasted_utility_alert": wasted_flag,
            "utility_message": utility_msg,
            "last_event": self.last_event,
            "event_log": self.event_log
        }
        
        return annotated_frame, telemetry
