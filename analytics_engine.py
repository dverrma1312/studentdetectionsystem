import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

class CampusAnalyticsEngine:
    def __init__(
        self,
        model_name="yolov8m.pt",
        conf_threshold=0.08,        # Ultra-low to catch every head/shoulder behind a desk
        blur_threshold=70.0,
        motion_threshold=0.015,
        light_threshold=140.0,
        line_position_ratio=0.15,
        imgsz=1280,
        **kwargs
    ):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        self.blur_threshold = blur_threshold
        self.motion_threshold = motion_threshold
        self.light_threshold = light_threshold
        self.line_position_ratio = line_position_ratio
        
        self.prev_gray = None
        self.all_seen_track_ids = set()
        self.active_tracks = {}
        self.track_history = defaultdict(list)
        self.track_postures = {}
        
        self.next_fallback_id = 1
        self.prev_centroids = {}
        
        self.event_log = []
        self.last_event = "System initialized"
        
    def _log_event(self, message):
        ts = time.strftime("%H:%M:%S")
        self.last_event = message
        self.event_log.insert(0, f"[{ts}] {message}")
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
                d = np.sqrt((cx-px)**2 + (cy-py)**2)
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

    def _enhance(self, frame):
        """CLAHE contrast boost — makes dark/shadowed people visible."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        crossing_x = int(w * self.line_position_ratio)

        blur_val, blur_status = self.check_blur(frame)
        motion_status, _ = self.detect_motion(frame)

        # ── SINGLE HIGH-RES PASS with contrast enhancement ──
        # No tiling = no lag. augment=True adds internal multi-scale + flip.
        enhanced = self._enhance(frame)

        try:
            results = self.model.track(
                source=enhanced,
                classes=[0],
                conf=self.conf_threshold,
                iou=0.3,          # Lower IoU = keep nearby seated students separate
                imgsz=self.imgsz,
                max_det=50,       # Allow up to 50 people per frame
                augment=True,     # Test-time augmentation (multi-scale + flip)
                persist=True,
                verbose=False
            )
        except Exception:
            results = self.model(
                source=enhanced, classes=[0], conf=self.conf_threshold,
                iou=0.3, imgsz=self.imgsz, max_det=50, augment=True, verbose=False
            )

        annotated = frame.copy()
        current_ids = set()
        seated, standing = 0, 0

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            
            if boxes.id is not None:
                tids = boxes.id.int().cpu().tolist()
            else:
                cents = [((int(b[0])+int(b[2]))//2, (int(b[1])+int(b[3]))//2) for b in xyxy]
                tids = self._assign_fallback_ids(cents)

            for i, box in enumerate(xyxy):
                tid = tids[i] if i < len(tids) else i + 1
                current_ids.add(tid)
                self.all_seen_track_ids.add(tid)

                x1, y1, x2, y2 = map(int, box)
                bw, bh = max(x2-x1, 1), max(y2-y1, 1)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                self.track_history[tid].append((cx, cy))
                if len(self.track_history[tid]) > 20:
                    self.track_history[tid].pop(0)

                hist = self.track_history[tid]
                vel = 0.0
                if len(hist) >= 5:
                    vel = np.sqrt((hist[-1][0]-hist[0][0])**2 + (hist[-1][1]-hist[0][1])**2)

                ar = bh / bw
                if vel < 15.0 or ar < 1.7:
                    posture, col = "Seated", (255, 191, 0)
                    seated += 1
                else:
                    posture, col = "Standing/Moving", (0, 255, 127)
                    standing += 1
                self.track_postures[tid] = posture

                # Vertical line crossing
                if tid not in self.active_tracks:
                    self.active_tracks[tid] = {
                        'last_x': cx,
                        'state': "inside" if cx > crossing_x else "outside"
                    }
                else:
                    prev = self.active_tracks[tid]['state']
                    if prev == "outside" and cx >= crossing_x:
                        self.active_tracks[tid]['state'] = "inside"
                        self._log_event(f"Entry detected (Person #{tid})")
                    elif prev == "inside" and cx < crossing_x:
                        self.active_tracks[tid]['state'] = "outside"
                        self._log_event(f"Exit detected (Person #{tid})")
                    self.active_tracks[tid]['last_x'] = cx

                # Draw
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                lbl = f"#{tid}|{posture}"
                (wl, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(annotated, (x1, y1-15), (x1+wl+2, y1), col, -1)
                cv2.putText(annotated, lbl, (x1+1, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                cv2.circle(annotated, (cx, cy), 3, (0,0,255), -1)

        n_present = len(current_ids)
        n_unique = len(self.all_seen_track_ids)

        wasted, util_msg = self.check_wasted_utility(frame, n_present)
        if wasted and "Lights ON" not in self.last_event:
            self._log_event("UTILITY ALERT: Lights left ON in empty room!")

        # Entry/exit vertical line
        cv2.line(annotated, (crossing_x, 0), (crossing_x, h), (0,0,255), 2)
        cv2.putText(annotated, "ENTRY/EXIT", (crossing_x+5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.putText(annotated, f"Detected: {n_present}", (w-250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

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
