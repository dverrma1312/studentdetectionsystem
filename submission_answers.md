# iCloudEMS Technical Interview Submission — Written Responses

### Question 1: How would you scale this from one live camera to 500 cameras streaming at once? Where would it break first?

**Answer:**
To scale to 500 live cameras, we would adopt an **Edge-to-Cloud Hybrid Architecture**. Instead of streaming 500 raw high-definition RTSP video streams over campus Wi-Fi/LAN to a centralized server (which will immediately saturate network bandwidth), we deploy lightweight edge gateways (such as NVIDIA Jetson devices or local campus micro-servers) running TensorRT-optimized YOLO models at each building cluster. The edge nodes process video frames locally and emit lightweight JSON telemetry payloads (person counts, occupancy events, timestamps) over MQTT or Apache Kafka to a centralized cloud broker. 

**Where it breaks first:** The architecture will break first at **centralized video stream ingestion bandwidth and CPU/GPU decoding bottlenecks**. Decoding 500 H.264/H.265 video streams concurrently on a single server exhausts hardware video decoders (NVDEC) long before GPU model inference capacity is maxed out. Edge processing and frame sampling (5–10 FPS) prevent this bottleneck.

---

### Question 2: How would you avoid double-counting or losing track of a person if they briefly leave the camera's view?

**Answer:**
We resolve brief occlusions or camera boundary exits by combining **Kalman Filter trajectory prediction** with **Deep Re-Identification (Re-ID) feature embeddings**. Standard ByteTrack maintains spatial bounding box momentum when a target is temporarily occluded behind an object or another person. If a person exits the camera view entirely for a short grace window (e.g., 5 to 10 seconds), our pipeline extracts a lightweight appearance feature vector (Re-ID embedding) of the individual's upper-body clothing and spatial trajectory vector upon exit. When a new detection occurs near the doorway boundary within the time-to-live (TTL) window, we compute cosine similarity between the candidate's embedding and the recently exited track history. If similarity exceeds threshold ($> 0.85$), the system re-assigns the original `track_id` instead of generating a new unique count.

---

### Question 3: How would you handle a camera feed that's consistently blurry or poor quality — flag it, skip it, or something else?

**Answer:**
We handle consistently low-quality feeds using a **Graduated Fallback & Auto-Flagging Strategy**. 

1. **Automated Quality Filtering & Enhancement**: First, frames are continuously audited using Laplacian Variance ($< 70.0$) and BRISQUE quality scores. Minor blur or low lighting triggers an automated hardware-accelerated CLAHE (Contrast Limited Adaptive Histogram Equalization) and sharpening pre-processing filter to attempt recovery.
2. **Graceful Pipeline Fallback**: If the feed remains severely degraded, the system dynamically downgrades from high-confidence object detection (YOLO) to a robust, lower-overhead **MOG2 Background Subtraction & Motion-Only Analytics Mode**.
3. **Automated Maintenance Alerting**: The system flags the camera ID in the campus ops dashboard as `NEEDS_MAINTENANCE` (indicating potential lens obstruction, focus drift, or dirty physical housing) and alerts facilities staff, while preserving basic occupancy motion logging rather than silently dropping or miscounting data.
