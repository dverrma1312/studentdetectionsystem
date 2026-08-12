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
        conf_threshold=0.08,
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

        # Load Haar cascades for supplementary head/upperbody detection
        # Use locally downloaded cascades (more reliable than cv2.data.haarcascades)
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
                        print(f"  ✅ Loaded {name}")
                    else:
                        print(f"  ⚠️ {name} loaded but empty")
                except Exception as e:
                    print(f"  ❌ {name} error: {e}")

        self.prev_gray = None
        self.all_seen_track_ids = set()
        self.active_tracks = {}
        self.track_history = defaultdict(list)
        self.track_postures = {}

        self.next_fallback_id = 1
        self.prev_centroids = {}

        self.event_log = []
        self.last_event = "System initialized"

    # ─── helpers ────────────────────────────────────────────
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

    def _enhance(self, frame):
        """CLAHE contrast boost."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _box_overlaps_any(self, box, existing_boxes, iou_thresh=0.2):
        """Check if a box significantly overlaps any existing box."""
        x1, y1, x2, y2 = box
        for ex in existing_boxes:
            ex1, ey1, ex2, ey2 = ex[:4]
            # Intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_new = (x2 - x1) * (y2 - y1)
            # If the new box is mostly inside an existing one, skip it
            if area_new > 0 and inter / area_new > iou_thresh:
                return True
        return False

    def _detect_missed_people(self, gray, existing_boxes):
        """
        Run Haar cascades (upper body + face) to find people YOLO missed.
        Returns list of [x1, y1, x2, y2] boxes that don't overlap YOLO detections.
        """
        extra_boxes = []

        for cascade, scale, neighbors, min_size in [
            (self.upper_cascade, 1.05, 3, (30, 30)),
            (self.face_cascade, 1.1, 4, (20, 20)),
        ]:
            if cascade is None:
                continue
            try:
                detections = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale,
                    minNeighbors=neighbors,
                    minSize=min_size,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            except Exception:
                continue

            if detections is None or len(detections) == 0:
                continue

            for (x, y, w, h) in detections:
                # Expand Haar box slightly to approximate person bounding box
                pad_x, pad_y = int(w * 0.3), int(h * 0.5)
                bx1 = max(0, x - pad_x)
                by1 = max(0, y - pad_y // 2)
                bx2 = x + w + pad_x
                by2 = y + h + pad_y
                
                new_box = [bx1, by1, bx2, by2]
                # Only add if it doesn't overlap an existing YOLO detection
                all_existing = list(existing_boxes) + extra_boxes
                if not self._box_overlaps_any(new_box, all_existing, iou_thresh=0.2):
                    extra_boxes.append(new_box)

        return extra_boxes

    # ─── main pipeline ─────────────────────────────────────
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        crossing_x = int(w * self.line_position_ratio)

        blur_val, blur_status = self.check_blur(frame)
        motion_status, _ = self.detect_motion(frame)

        enhanced = self._enhance(frame)

        # ── Detector 1: YOLO (fast single pass) ──
        try:
            results = self.model.track(
                source=enhanced, classes=[0], conf=self.conf_threshold,
                iou=0.3, imgsz=self.imgsz, max_det=50,
                persist=True, verbose=False
            )
        except Exception:
            results = self.model(
                source=enhanced, classes=[0], conf=self.conf_threshold,
                iou=0.3, imgsz=self.imgsz, max_det=50, verbose=False
            )

        # Collect YOLO boxes
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

        # ── Detector 2: Haar cascades for missed people ──
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        haar_extra = self._detect_missed_people(gray, yolo_boxes)

        # Assign IDs to Haar detections
        if haar_extra:
            haar_cents = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in haar_extra]
            haar_tids = self._assign_fallback_ids(haar_cents)
        else:
            haar_tids = []

        # ── Merge all detections ──
        all_boxes = yolo_boxes + haar_extra
        all_tids = yolo_tids + haar_tids

        annotated = frame.copy()
        current_ids = set()
        seated, standing = 0, 0

        for i, box in enumerate(all_boxes):
            tid = all_tids[i] if i < len(all_tids) else i + 1
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
                vel = np.sqrt((hist[-1][0] - hist[0][0]) ** 2 +
                              (hist[-1][1] - hist[0][1]) ** 2)

            ar = bh / bw
            if vel < 15.0 or ar < 1.7:
                posture, col = "Seated", (255, 191, 0)
                seated += 1
            else:
                posture, col = "Standing/Moving", (0, 255, 127)
                standing += 1
            self.track_postures[tid] = posture

            # Is this a Haar-only detection?
            is_haar = i >= len(yolo_boxes)
            if is_haar:
                col = (0, 200, 255)  # Orange for Haar-supplemented detections

            # Vertical line crossing on LEFT
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
            src = "H" if is_haar else "Y"
            lbl = f"#{tid}|{posture}|{src}"
            (wl, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(annotated, (x1, y1 - 15), (x1 + wl + 2, y1), col, -1)
            cv2.putText(annotated, lbl, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        n_present = len(current_ids)
        n_unique = len(self.all_seen_track_ids)

        wasted, util_msg = self.check_wasted_utility(frame, n_present)
        if wasted and "Lights ON" not in self.last_event:
            self._log_event("UTILITY ALERT: Lights left ON in empty room!")

        # Draw vertical entry/exit line
        cv2.line(annotated, (crossing_x, 0), (crossing_x, h), (0, 0, 255), 2)
        cv2.putText(annotated, "ENTRY/EXIT", (crossing_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Count overlay
        yolo_n = len(yolo_boxes)
        haar_n = len(haar_extra)
        cv2.putText(annotated, f"Detected: {n_present} (YOLO:{yolo_n} + Haar:{haar_n})",
                    (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
