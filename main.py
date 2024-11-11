import cv2
import torch
import clip
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration

def extract_frames(video_path, frame_rate=1):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    
    while success:
        if count % frame_rate == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_features(frames):
    frame_features = []
    for frame in frames:
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        with torch.no_grad():
            frame_features.append(model.encode_image(image))
    return frame_features

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_descriptions(descriptions):
    input_text = " ".join(descriptions)  # Combine all descriptions
    inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)