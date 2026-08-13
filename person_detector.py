import torch
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name="yolov8s.pt"):
        """
        Unified Person & Pose Detector.
        Automatically detects available hardware acceleration (MPS/Metal for Apple, CUDA for NVIDIA, CPU).
        """
        self.model_name = model_name
        
        # Hardware Acceleration autodetect
        if torch.backends.mps.is_available():
            self.device = "mps"      # Apple Silicon GPU & Neural Engine
        elif torch.cuda.is_available():
            self.device = "cuda"     # NVIDIA GPU
        else:
            self.device = "cpu"      # Fallback CPU
            
        print(f"🚀 Initializing model '{model_name}' on hardware accelerator: '{self.device}'")
        self.model = YOLO(model_name)

    def predict(self, frame, conf=0.10, imgsz=640):
        """
        Runs object detection/tracking on a single frame.
        """
        # If model is a pose model, it will extract keypoints
        is_pose = "pose" in self.model_name
        
        try:
            results = self.model.track(
                source=frame,
                classes=[0] if not is_pose else None, # Class 0 is Person in COCO (YOLO pose only tracks keypoints)
                conf=conf,
                imgsz=imgsz,
                device=self.device,
                persist=True,
                verbose=False
            )
        except Exception:
            # Fallback to standard detect if tracking fails
            results = self.model(
                source=frame,
                classes=[0] if not is_pose else None,
                conf=conf,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )
        return results
