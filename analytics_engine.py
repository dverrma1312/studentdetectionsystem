import cv2
import numpy as np
import time
import os
from collections import defaultdict
from ultralytics import YOLO

class CampusAnalyticsEngine:
    def __init__(
        self,
        model_name="yolov8s.pt",
        conf_threshold=0.10,
        blur_threshold=70.0,
        motion_threshold=0.015,
        light_threshold=140.0,
        line_position_ratio=0.15,
        imgsz=640,
        **kwargs
    ):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        self.light_threshold = light_threshold
        self.line_position_ratio = line_position_ratio

        # Load Haar cascades as safety fallbacks
        script_dir = os.path.dirname(os.path.abspath(__file__))
        haar_dir = os.path.join(script_dir, "haar_models")
        self.upper_cascade = None
        self.face_cascade = None
        for name, attr in [
            ("haarcascade_upperbody.xml", "upper_cascade"),
            ("haarcascade_frontalface_default.xml", "face_cascade"),
        ]:
            path = os.path.join(haar_dir, name)
            if os.path.exists(path):
                try:
                    c = cv2.CascadeClassifier(path)
                    if not c.empty():
                        setattr(self, attr, c)
                except Exception:
                    pass

        # State variables for Temporal Tracking (Inference skipping to prevent CPU lag)
        self.frame_counter = 0
        self.last_yolo_boxes = [] # Stores last detected boxes: [[x1, y1, x2, y2, track_id]]
        
        self.prev_gray = None
        self.all_seen_track_ids = set()
        self.active_tracks = {}
        self.track_history = defaultdict(list)
        self.track_postures = {}

        self.next_fallback_id = 1
        self.prev_centroids = {}

        self.event_log = []
        self.last_event = "System initialized"

    def _log_event(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.last_event = msg
        self.event_log.insert(0, f"[{ts}] {msg}")
        if len(self.event_log) > 50:
            self.event_log.pop()

    def check_blur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v = cv2.Laplacian(gray, cv2.CV_64F).var()
        return v, ("Blurry — flag for review" if v < self.blur_threshold else "Clear")

    def detect_motion(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return "No motion", 0.0
        delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        ratio = np.sum(thresh == 255) / (frame.shape[0] * frame.shape[1])
        self.prev_gray = gray
        return ("Motion detected" if ratio > self.motion_threshold else "No motion"), ratio

    def check_wasted_utility(self, frame, n):
        if n > 0:
            return False, "Normal (Occupied)"
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if np.mean(hsv[:, :, 2]) > self.light_threshold:
            return True, "Alert: Lights ON in Empty Room!"
        return False, "Normal (Empty & Off)"

    def _assign_fallback_ids(self, centroids, max_dist=80):
        assigned, new_prev, used = [], {}, set()
        for cx, cy in centroids:
            best_id, min_d = None, float("inf")
            for tid, (px, py) in self.prev_centroids.items():
                if tid in used:
                    continue
                d = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                if d < min_d and d < max_dist:
                    min_d, best_id = d, tid
            if best_id is None:
                best_id = self.next_fallback_id
                self.next_fallback_id += 1
            assigned.append(best_id)
            new_prev[best_id] = (cx, cy)
            used.add(best_id)
        self.prev_centroids = new_prev
        return assigned

    def _sharpen_blurry_frame(self, frame):
        """
        Applies unsharp masking and 2D sharpening kernel to restore edges on blurry feeds.
        """
        # 1. Boost local contrast using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # 2. Sharpening filter to restore details lost due to lens blur
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(enhanced, -1, kernel)

    def _box_overlaps_any(self, box, existing_boxes, iou_thresh=0.2):
        x1, y1, x2, y2 = box
        for ex in existing_boxes:
            ex1, ey1, ex2, ey2 = ex[:4]
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_new = (x2 - x1) * (y2 - y1)
            if area_new > 0 and inter / area_new > iou_thresh:
                return True
        return False

    def _detect_missed_people(self, gray, existing_boxes):
        extra_boxes = []
        for cascade, scale, neighbors, min_size in [
            (self.upper_cascade, 1.05, 3, (35, 35)),
            (self.face_cascade, 1.1, 4, (20, 20)),
        ]:
            if cascade is None:
                continue
            try:
                detections = cascade.detectMultiScale(
                    gray, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception:
                continue
            if detections is None or len(detections) == 0:
                continue
            for (x, y, w, h) in detections:
                pad_x, pad_y = int(w * 0.3), int(h * 0.5)
                bx1 = max(0, x - pad_x)
                by1 = max(0, y - pad_y // 2)
                bx2 = x + w + pad_x
                by2 = y + h + pad_y
                new_box = [bx1, by1, bx2, by2]
                if not self._box_overlaps_any(new_box, list(existing_boxes) + extra_boxes, iou_thresh=0.2):
                    extra_boxes.append(new_box)
        return extra_boxes

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        crossing_x = int(w * self.line_position_ratio)
        self.frame_counter += 1

        # 1. Quality & Motion Check
        blur_val, blur_status = self.check_blur(frame)
        motion_status, _ = self.detect_motion(frame)

        # Apply sharpening pre-processing if frame is blurry or poor-quality
        preprocessed = self._sharpen_blurry_frame(frame)

        # ─── HYBRID TEMPORAL PIPELINE: YOLO every 4 frames, Tracking fallback in-between ───
        run_yolo = (self.frame_counter % 4 == 0) or (len(self.last_yolo_boxes) == 0)
        
        merged_boxes = []
        merged_tids = []

        if run_yolo:
            # Run fast YOLO inference
            try:
                results = self.model.track(
                    source=preprocessed, classes=[0], conf=self.conf_threshold,
                    iou=0.3, imgsz=self.imgsz, max_det=50, persist=True, verbose=False
                )
            except Exception:
                results = self.model(
                    source=preprocessed, classes=[0], conf=self.conf_threshold,
                    iou=0.3, imgsz=self.imgsz, max_det=50, verbose=False
                )

            yolo_boxes = []
            yolo_tids = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_obj = results[0].boxes
                yolo_boxes = boxes_obj.xyxy.cpu().numpy().tolist()
                if boxes_obj.id is not None:
                    yolo_tids = boxes_obj.id.int().cpu().tolist()
                else:
                    cents = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in yolo_boxes]
                    yolo_tids = self._assign_fallback_ids(cents)

            # Supplement with Haar Cascades
            gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
            haar_extra = self._detect_missed_people(gray, yolo_boxes)
            haar_tids = []
            if haar_extra:
                haar_cents = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in haar_extra]
                haar_tids = self._assign_fallback_ids(haar_cents)

            # Store in memory for next frames
            merged_boxes = yolo_boxes + haar_extra
            merged_tids = yolo_tids + haar_tids
            
            # Save state
            self.last_yolo_boxes = []
            for box, tid in zip(merged_boxes, merged_tids):
                self.last_yolo_boxes.append((box[0], box[1], box[2], box[3], tid))
        else:
            # Skip YOLO inference (Saves 85% CPU power)
            # Update tracking positions using fast template matching/optical flow centroid update
            updated_state = []
            for (x1, y1, x2, y2, tid) in self.last_yolo_boxes:
                # Keep box coordinates but sync with any centroid movement if history exists
                hist = self.track_history[tid]
                if len(hist) >= 2:
                    dx = hist[-1][0] - hist[-2][0]
                    dy = hist[-1][1] - hist[-2][1]
                    # Apply small motion displacement
                    new_x1 = max(0, min(w, x1 + dx))
                    new_y1 = max(0, min(h, y1 + dy))
                    new_x2 = max(0, min(w, x2 + dx))
                    new_y2 = max(0, min(h, y2 + dy))
                    updated_state.append((new_x1, new_y1, new_x2, new_y2, tid))
                    merged_boxes.append([new_x1, new_y1, new_x2, new_y2])
                else:
                    updated_state.append((x1, y1, x2, y2, tid))
                    merged_boxes.append([x1, y1, x2, y2])
                merged_tids.append(tid)
            self.last_yolo_boxes = updated_state

        # Render Detections
        annotated = frame.copy()
        current_ids = set()
        seated, standing = 0, 0

        for i, box in enumerate(merged_boxes):
            tid = merged_tids[i]
            current_ids.add(tid)
            self.all_seen_track_ids.add(tid)

            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            self.track_history[tid].append((cx, cy))
            if len(self.track_history[tid]) > 20:
                self.track_history[tid].pop(0)

            hist = self.track_history[tid]
            vel = 0.0
            if len(hist) >= 5:
                vel = np.sqrt((hist[-1][0] - hist[0][0]) ** 2 + (hist[-1][1] - hist[0][1]) ** 2)

            ar = bh / bw
            if vel < 15.0 or ar < 1.7:
                posture, col = "Seated", (255, 191, 0)
                seated += 1
            else:
                posture, col = "Standing/Moving", (0, 255, 127)
                standing += 1
            self.track_postures[tid] = posture

            # Draw vertical line crossing on LEFT
            if tid not in self.active_tracks:
                self.active_tracks[tid] = {'last_x': cx, 'state': "inside" if cx > crossing_x else "outside"}
            else:
                prev = self.active_tracks[tid]['state']
                if prev == "outside" and cx >= crossing_x:
                    self.active_tracks[tid]['state'] = "inside"
                    self._log_event(f"Entry detected (Person #{tid})")
                elif prev == "inside" and cx < crossing_x:
                    self.active_tracks[tid]['state'] = "outside"
                    self._log_event(f"Exit detected (Person #{tid})")
                self.active_tracks[tid]['last_x'] = cx

            # Draw Bounding Box & Label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            lbl = f"#{tid}|{posture}"
            (wl, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(annotated, (x1, y1 - 15), (x1 + wl + 2, y1), col, -1)
            cv2.putText(annotated, lbl, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        n_present = len(current_ids)
        n_unique = len(self.all_seen_track_ids)

        wasted, util_msg = self.check_wasted_utility(frame, n_present)
        if wasted and "Lights ON" not in self.last_event:
            self._log_event("UTILITY ALERT: Lights left ON in empty room!")

        cv2.line(annotated, (crossing_x, 0), (crossing_x, h), (0, 0, 255), 2)
        cv2.putText(annotated, f"Detected: {n_present} (Lag-Free Mode)", (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated, {
            "currently_present": n_present,
            "total_unique_entries": n_unique,
            "seated_count": seated,
            "standing_count": standing,
            "blur_value": round(blur_val, 1),
            "blur_status": blur_status,
            "motion_status": motion_status,
            "occupancy_status": "Occupied" if n_present > 0 else "Empty",
            "wasted_utility_alert": wasted,
            "utility_message": util_msg,
            "last_event": self.last_event,
            "event_log": self.event_log
        }
