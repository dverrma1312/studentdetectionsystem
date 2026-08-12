import cv2
import numpy as np

def create_synthetic_stream(output_path="sample_stream.mp4", duration_sec=15, fps=30):
    """
    Generates a synthetic CCTV video feed showing people entering, sitting, moving, and exiting.
    Useful for instant testing without a webcam.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = duration_sec * fps
    
    # Person 1 path: Enter from top (y=50), cross line (y=240), sit down, stand up, exit back to top
    # Person 2 path: Enter from top (y=20), cross line (y=240), walk around, sit down

    for frame_idx in range(num_frames):
        # Create room background (grayish classroom background)
        frame = np.full((height, width, 3), (220, 220, 220), dtype=np.uint8)
        
        # Draw some classroom furniture (desks/chairs)
        cv2.rectangle(frame, (100, 320), (220, 400), (160, 160, 160), -1) # Desk 1
        cv2.putText(frame, "Desk 1", (130, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        cv2.rectangle(frame, (400, 320), (520, 400), (160, 160, 160), -1) # Desk 2
        cv2.putText(frame, "Desk 2", (430, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

        # Draw Doorway indicator
        cv2.rectangle(frame, (250, 0), (390, 40), (100, 100, 100), -1)
        cv2.putText(frame, "ENTRANCE", (275, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        progress = frame_idx / num_frames

        # Person 1 movement trajectory
        if progress < 0.25:
            # Entering (moving down)
            p1_y = int(50 + (progress / 0.25) * 280)
            p1_x = 160
            p1_h, p1_w = 120, 50 # Standing
        elif progress < 0.60:
            # Seated at Desk 1
            p1_y = 330
            p1_x = 160
            p1_h, p1_w = 60, 60 # Seated (shorter height)
        else:
            # Standing up and Exiting (moving up)
            p1_y = int(330 - ((progress - 0.60) / 0.40) * 300)
            p1_x = 160
            p1_h, p1_w = 120, 50 # Standing

        # Draw Person 1 (Blue silhouette)
        cv2.ellipse(frame, (p1_x, p1_y - p1_h + 15), (15, 15), 0, 0, 360, (200, 50, 50), -1) # Head
        cv2.rectangle(frame, (p1_x - p1_w//2, p1_y - p1_h + 30), (p1_x + p1_w//2, p1_y), (200, 50, 50), -1) # Body

        # Person 2 movement trajectory (Enters at progress 0.20)
        if progress > 0.20:
            p2_progress = (progress - 0.20) / 0.80
            if p2_progress < 0.4:
                p2_y = int(30 + (p2_progress / 0.4) * 300)
                p2_x = 460
                p2_h, p2_w = 120, 50 # Standing
            else:
                p2_y = 330
                p2_x = 460
                p2_h, p2_w = 65, 60 # Seated

            # Draw Person 2 (Red/Orange silhouette)
            cv2.ellipse(frame, (p2_x, p2_y - p2_h + 15), (15, 15), 0, 0, 360, (50, 50, 200), -1)
            cv2.rectangle(frame, (p2_x - p2_w//2, p2_y - p2_h + 30), (p2_x + p2_w//2, p2_y), (50, 50, 200), -1)

        # Simulate blur at progress 0.80 for 1 second
        if 0.78 <= progress <= 0.85:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)

        out.write(frame)

    out.release()
    print(f"Synthetic video generated at '{output_path}'")

if __name__ == "__main__":
    create_synthetic_stream()
