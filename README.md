# iCloudEMS Campus Intelligence — Student Detection & Analytics System

Real-Time CCTV Stream Analytics Pipeline for Automated Attendance, Room Occupancy, Posture Classification, Frame Quality Auditing, and Energy Optimization.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8%20%2B%20ByteTrack-green)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📌 Features

- **Person Detection & Persistent Tracking**: Powered by YOLOv8 and ByteTrack to track unique individuals with persistent IDs (`Person #1`, `Person #2`).
- **Entry & Exit Counter**: Trajectory tracking across configurable boundary lines to record cumulative **Total Unique Entries** and **Currently Present**.
- **Seated vs. Standing/Moving Classification**: Velocity and aspect-ratio heuristics to classify posture in crowded classrooms (`Seated` vs `Standing/Moving`).
- **Frame Quality Auditor**: Laplacian Variance calculation to detect camera blur (`Clear` vs `Blurry — flag for review`).
- **Motion Detection**: Grayscale frame differencing (`Motion detected` vs `No motion`).
- **Wasted Energy Utility Alerts**: Detects when room lights are left ON while room occupancy is 0 (`Lights ON in Empty Room`).
- **Interactive Web Dashboard**: Streamlit-based UI with live stream overlays, real-time KPI metrics, scrolling event logs, and confidence sliders.

---

## 🏗️ Architecture

```
Live Camera Feed / Video Stream -> Frame Quality & Motion Audit -> YOLOv8 + ByteTrack -> Line Crossing & Posture Engine -> Streamlit Telemetry Dashboard
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/dverrma1312/studentdetectionsystem.git
cd studentdetectionsystem

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
├── analytics_engine.py      # Core computer vision processing pipeline
├── app.py                   # Streamlit live dashboard web app
├── generate_test_video.py   # Synthetic CCTV video feed generator
├── test_pipeline.py         # Automated pipeline test runner
├── submission_answers.md    # Scaling, occlusion & blur technical Q&A
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 📄 Technical Q&A Submission Answers

See [`submission_answers.md`](submission_answers.md) for full answers on:
1. Scaling to 500 concurrent live cameras (Edge-to-Cloud hybrid architecture).
2. Avoiding double-counting during brief occlusions/re-entries (Kalman Filter + Deep Re-ID embeddings).
3. Handling blurry/poor-quality feeds (CLAHE enhancement, MOG2 fallback, auto-flagging for maintenance).
