import cv2
import os
import numpy as np
from datetime import datetime, timezone
import shutil

# --- Video input and output folder ---
video_path = '/home/nvp/Downloads/raphson.mp4'
output_dir = '/home/nvp/ros2_test/src/ros2_orb_slam3/TEST_DATASET/sample_euroc_MH05/mav0/cam0/data'

# Clear output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# --- Get timestamp anchor ---
now = datetime.now(timezone.utc)
seconds_since_midnight = (
    now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1_000_000
)
start_time_ns = int(seconds_since_midnight * 1_000_000_000)

# --- Frame loop ---
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (no undistortion)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Generate timestamp in nanoseconds
    frame_offset_ns = int((frame_idx / fps) * 1_000_000_000)
    timestamp_ns = start_time_ns + frame_offset_ns

    # Save
    filename = os.path.join(output_dir, f'{timestamp_ns}.png')
    cv2.imwrite(filename, gray_frame)

    frame_idx += 1

cap.release()
print(f"Saved {frame_idx} frames to '{output_dir}' with UTC time in nanoseconds.")
# print("Sample grayscale frame shape:", gray_frame.shape)

