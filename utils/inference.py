import os
import sys
import torch
import cv2
import numpy as np

# --- THIS IS THE FIX ---
# Import the ResNet+LSTM model, not the VideoResNet
from yes_yes_integration.models.cnn import resnet18_lstm_model

def load_model(model_path):
    """
    Loads a pre-trained ResNet+LSTM model from a .pth file.
    """
    
    # --- THIS IS THE FIX ---
    # The 'resnet18_lstm_model' function expects a config object 'c'
    # with specific parameters found in 'models/cnn.py' and 'train_video_resnet.py'
    #.
    class SimpleConfig:
        # --- IMPORTANT ---
        # Your 'app.py' is set up for 5 classes (in CLASS_NAMES)
        # Your 'train_video_resnet.py' shows NUM_CLASSES = 14.
        # MAKE SURE this number matches the model you trained!
        # If your trained model had 14 classes, change this to 14
        # and update CLASS_NAMES in 'app.py'.
        NUM_CLASSES = 5 
        
        # Default values from your project files
        LSTM_HIDDEN_DIM = 128
        LSTM_NUM_LAYERS = 1
        LSTM_DROPOUT = 0.4
        FC_DROPOUT = 0.5
        
    model = resnet18_lstm_model(SimpleConfig) # Call the correct model function
    # ---------------------

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_video(video_path, num_frames=32, img_size=224):
    """
    Reads a video file, samples frames, and preprocesses them into a tensor.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Could not read video file: {video_path}")

    # Ensure indices are valid
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If reading fails, try to grab the next available frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx + 1)
            ret, frame = cap.read()
            if not ret:
                continue # Skip if frame is still not readable
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (img_size, img_size))
        # Normalize to [0, 1] and permute to (C, H, W)
        frame = torch.tensor(frame / 255.0).permute(2, 0, 1).float()
        frames.append(frame)

    cap.release()
    
    if not frames:
        raise ValueError("Could not extract any frames from the video.")

    # Stack frames and permute to (B, T, C, H, W)
    # The ResNet+LSTM model expects this shape
    video_tensor = torch.stack(frames).unsqueeze(0) # (1, num_frames, 3, H, W)
    
    return video_tensor

def predict_video(model, video_path, class_names):
    """
    Performs end-to-end inference on a single video file.
    """
    # Preprocess the video
    video_tensor = preprocess_video(video_path)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probs, dim=1)
        
        confidence = top_prob.item()
        predicted_class = class_names[top_class_idx.item()]
        
    return predicted_class, confidence