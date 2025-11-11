# utils/inference.py
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

# Load model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Example: 5 classes
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Preprocess frames
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(frame)

# Inference on a video
def predict_video(model, video_path, class_names):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # sample frames evenly
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess_frame(frame))
    
    cap.release()
    
    if not frames:
        return "Error: No frames read"
    
    batch = torch.stack(frames)
    with torch.no_grad():
        preds = model(batch)
        avg_pred = torch.mean(preds, dim=0)
        label_idx = torch.argmax(avg_pred).item()
        confidence = torch.softmax(avg_pred, dim=0)[label_idx].item()
    
    return class_names[label_idx], confidence
