import cv2
import os

video_path = "video/v.mp4"
output_folder = "sampled_frames"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate interval to sample 30 frames
frame_interval = total_frames // 30

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save every 'frame_interval'-th frame
    if frame_count % frame_interval == 0 and saved_count < 30:
        output_path = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Total frames saved: {saved_count}")
print(f"Frames saved in folder: {output_folder}")
