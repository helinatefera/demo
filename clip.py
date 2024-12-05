import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to the folder with the 30 frames
frames_folder = "sampled_frames"
frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".jpg")])

# Placeholder to store descriptions
frame_descriptions = []

# Loop through each frame and generate description
for frame_path in frames:
    image = Image.open(frame_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    # Encode image
    image_features = model.get_image_features(**inputs)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Optional: Create a textual prompt space (e.g., "A scene with...")
    prompts = ["A scenic view", "A person walking", "An outdoor area", "An indoor scene", "A car on the road"]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    # Extract image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Compute similarity
    similarities = torch.matmul(image_features, text_features.T)
    best_prompt_idx = similarities.argmax().item()
    description = prompts[best_prompt_idx]
    
    # Store description
    frame_descriptions.append((frame_path, description))
    print(f"Frame: {frame_path}, Description: {description}")

# Print or save the descriptions
for frame, description in frame_descriptions:
    print(f"{frame}: {description}")
