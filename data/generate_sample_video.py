import os
import cv2
import numpy as np

def generate_sample_video(output_path: str = "data/videos/sample_surveillance.mp4", duration_sec: int = 10, fps: int = 30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.rectangle(frame, (400, 200), (880, 600), (0, 0, 150), 2)
        cv2.putText(frame, "RESTRICTED ZONE A", (410, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if 60 <= frame_idx <= 240:
            progress = (frame_idx - 60) / 180.0
            x = int(100 + progress * 800)
            y = int(350 + np.sin(progress * np.pi * 2) * 50)
            cv2.circle(frame, (x + 30, y - 20), 20, (200, 180, 150), -1)
            cv2.rectangle(frame, (x, y), (x + 60, y + 140), (180, 100, 50), -1)
            cv2.putText(frame, "Person Sim", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if 150 <= frame_idx <= 280:
            progress = (frame_idx - 150) / 130.0
            vx = int(1200 - progress * 1000)
            vy = 620
            cv2.rectangle(frame, (vx, vy), (vx + 150, vy + 70), (50, 150, 220), -1)
            cv2.putText(frame, "Vehicle Sim", (vx + 10, vy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"Sample video generated at {output_path}")

if __name__ == "__main__":
    generate_sample_video()
