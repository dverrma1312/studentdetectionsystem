import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

class CampusAnalyticsEngine:
    def __init__(
        self,
        model_name="yolov8s.pt",
        conf_threshold=0.12,
        blur_threshold=70.0,
        motion_threshold=0.015,
        light_threshold=140.0,
        line_position_ratio=0.15,   # Vertical line on left side of screen (15% from left)
        imgsz=1280,                 # Ultra high-res for maximum person detection
        **kwargs
    ):
        """
        Campus Intelligence Analytics Engine.
        Entry/Exit line is VERTICAL on the LEFT side of the frame (like a doorway).
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        self.light_threshold = light_threshold
        self.line_position_ratio = line_position_ratio
        
        # State
        self.prev_gray = None
        self.all_seen_track_ids = set()
        self.active_tracks = {}       # track_id -> {'last_x': int, 'state': 'outside'/'inside'}
        self.track_history = defaultdict(list)
        self.track_postures = {}
        
        # Fallback centroid tracker
        self.next_fallback_id = 1
        self.prev_centroids = {}
        
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < self.blur_threshold:
            return variance, "Blurry — flag for review"
        return variance, "Clear"

    def detect_motion(self, frame):
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
        if currently_present > 0:
            return False, "Normal (Occupied)"
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:, :, 2])
        if avg_brightness > self.light_threshold:
            return True, "Alert: Lights ON in Empty Room!"
        return False, "Normal (Empty & Off)"

    def _assign_fallback_ids(self, centroids, max_distance=80):
        assigned_ids = []
        new_prev = {}
        used_ids = set()
        for cx, cy in centroids:
            best_id = None
            min_dist = float("inf")
            for tid, (px, py) in self.prev_centroids.items():
                if tid in used_ids:
                    continue
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self.next_fallback_id
                self.next_fallback_id += 1
            assigned_ids.append(best_id)
            new_prev[best_id] = (cx, cy)
            used_ids.add(best_id)
        self.prev_centroids = new_prev
        return assigned_ids

    def _run_tiled_detection(self, frame):
        """
        Run YOLO on full frame AND on overlapping tiles, then merge all detections.
        This catches small/far people that a single pass at any resolution would miss.
        """
        h, w = frame.shape[:2]
        all_boxes = []
        all_confs = []

        # Pass 1: Full frame at high resolution
        r1 = self.model(
            source=frame, classes=[0], conf=self.conf_threshold,
            imgsz=self.imgsz, verbose=False
        )
        if r1 and r1[0].boxes is not None and len(r1[0].boxes) > 0:
            for b, c in zip(r1[0].boxes.xyxy.cpu().numpy(), r1[0].boxes.conf.cpu().numpy()):
                all_boxes.append(b)
                all_confs.append(c)

        # Pass 2: Overlapping tiles (left half, right half, top half, bottom half)
        tiles = [
            frame[0:h, 0:w//2+50],          # Left half + overlap
            frame[0:h, w//2-50:w],           # Right half + overlap
            frame[0:h//2+50, 0:w],           # Top half + overlap
            frame[h//2-50:h, 0:w],           # Bottom half + overlap
        ]
        offsets = [
            (0, 0),
            (w//2-50, 0),
            (0, 0),
            (0, h//2-50),
        ]

        for tile, (ox, oy) in zip(tiles, offsets):
            if tile.shape[0] < 50 or tile.shape[1] < 50:
                continue
            rt = self.model(
                source=tile, classes=[0], conf=self.conf_threshold,
                imgsz=max(640, self.imgsz // 2), verbose=False
            )
            if rt and rt[0].boxes is not None and len(rt[0].boxes) > 0:
                for b, c in zip(rt[0].boxes.xyxy.cpu().numpy(), rt[0].boxes.conf.cpu().numpy()):
                    # Remap tile coordinates back to full frame
                    all_boxes.append([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy])
                    all_confs.append(c)

        if not all_boxes:
            return np.array([]), np.array([])

        # NMS to remove duplicate detections from overlapping tiles
        boxes_np = np.array(all_boxes, dtype=np.float32)
        confs_np = np.array(all_confs, dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(
            bboxes=[(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes_np],
            scores=confs_np.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=0.45
        )
        if len(indices) == 0:
            return np.array([]), np.array([])
        
        indices = np.array(indices).flatten()
        return boxes_np[indices], confs_np[indices]

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        # VERTICAL line on the LEFT side of screen
        crossing_x = int(w * self.line_position_ratio)
        
        # 1. Blur check
        blur_val, blur_status = self.check_blur(frame)
        
        # 2. Motion check
        motion_status, motion_ratio = self.detect_motion(frame)
        
        # 3. Detection + Tracking
        annotated_frame = frame.copy()
        current_tracked_ids = set()
        seated_count = 0
        standing_count = 0
        
        # Try tracker first; fall back to tiled detection
        use_tracker = True
        try:
            results = self.model.track(
                source=frame, classes=[0], conf=self.conf_threshold,
                imgsz=self.imgsz, persist=True, verbose=False
            )
        except Exception:
            use_tracker = False
        
        xyxy_boxes = np.array([])
        track_ids = []
        
        if use_tracker and results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_obj = results[0].boxes
            xyxy_boxes = boxes_obj.xyxy.cpu().numpy()
            if boxes_obj.id is not None:
                track_ids = boxes_obj.id.int().cpu().tolist()
            else:
                centroids = [((int(b[0])+int(b[2]))//2, (int(b[1])+int(b[3]))//2) for b in xyxy_boxes]
                track_ids = self._assign_fallback_ids(centroids)
        
        # If tracker found very few, supplement with tiled detection
        if len(xyxy_boxes) < 5:
            tiled_boxes, tiled_confs = self._run_tiled_detection(frame)
            if len(tiled_boxes) > len(xyxy_boxes):
                xyxy_boxes = tiled_boxes
                centroids = [((int(b[0])+int(b[2]))//2, (int(b[1])+int(b[3]))//2) for b in xyxy_boxes]
                track_ids = self._assign_fallback_ids(centroids)

        # Process each detected person
        for i, box in enumerate(xyxy_boxes):
            track_id = track_ids[i] if i < len(track_ids) else i + 1
            current_tracked_ids.add(track_id)
            self.all_seen_track_ids.add(track_id)
            
            x1, y1, x2, y2 = map(int, box)
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Track history
            self.track_history[track_id].append((center_x, center_y))
            if len(self.track_history[track_id]) > 20:
                self.track_history[track_id].pop(0)
                
            # Velocity
            history = self.track_history[track_id]
            velocity = 0.0
            if len(history) >= 5:
                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                velocity = np.sqrt(dx**2 + dy**2)
                
            # Posture
            aspect_ratio = box_h / box_w
            if velocity < 15.0 or aspect_ratio < 1.7:
                posture = "Seated"
                seated_count += 1
                box_color = (255, 191, 0)
            else:
                posture = "Standing/Moving"
                standing_count += 1
                box_color = (0, 255, 127)
            self.track_postures[track_id] = posture
            
            # Line Crossing (VERTICAL line on LEFT — X axis)
            # Person moves from left-of-line to right-of-line = ENTRY
            # Person moves from right-of-line to left-of-line = EXIT
            if track_id not in self.active_tracks:
                initial_state = "inside" if center_x > crossing_x else "outside"
                self.active_tracks[track_id] = {'last_x': center_x, 'state': initial_state}
            else:
                prev_state = self.active_tracks[track_id]['state']
                if prev_state == "outside" and center_x >= crossing_x:
                    self.active_tracks[track_id]['state'] = "inside"
                    self._log_event(f"Entry detected (Person #{track_id})")
                elif prev_state == "inside" and center_x < crossing_x:
                    self.active_tracks[track_id]['state'] = "outside"
                    self._log_event(f"Exit detected (Person #{track_id})")
                self.active_tracks[track_id]['last_x'] = center_x
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"#{track_id}|{posture}"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 18), (x1 + w_lbl + 4, y1), box_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)

        # Metrics
        currently_present = len(current_tracked_ids)
        total_unique_entries = len(self.all_seen_track_ids)
        occupancy_status = "Occupied" if currently_present > 0 else "Empty"
        
        wasted_flag, utility_msg = self.check_wasted_utility(frame, currently_present)
        if wasted_flag and "Lights ON" not in self.last_event:
            self._log_event("UTILITY ALERT: Lights left ON in empty room!")
            
        # Draw VERTICAL Entry/Exit line on LEFT side
        cv2.line(annotated_frame, (crossing_x, 0), (crossing_x, h), (0, 0, 255), 2)
        cv2.putText(annotated_frame, "ENTRY/EXIT", (crossing_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(annotated_frame, "LINE", (crossing_x + 5, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw detection count on frame
        count_label = f"Detected: {currently_present} people"
        cv2.putText(annotated_frame, count_label, (w - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        telemetry = {
            "currently_present": currently_present,
            "total_unique_entries": total_unique_entries,
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
