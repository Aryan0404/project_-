# app.py
import streamlit as st
import tempfile
import os
from utils.inference import load_model, predict_video

# Initialize model once
@st.cache_resource
def get_model():
    model = load_model("model/resnet_model.pth")
    return model

st.title("Cricket Event Video Classification System")
st.write("Upload one or multiple videos, and the model will classify them.")

uploaded_videos = st.file_uploader(
    "Upload videos", type=["mp4", "avi", "mov"], accept_multiple_files=True
)

if uploaded_videos:
    model = get_model()
    class_names = ["Dance", "Football", "Cooking", "Driving", "Singing"]  # Example
    
    for uploaded_video in uploaded_videos:
        st.write(f"**File:** {uploaded_video.name}")
        
        # Save temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Run prediction
        label, confidence = predict_video(model, tfile.name, class_names)
        
        st.video(uploaded_video)
        st.success(f"**Predicted Class:** {label} (Confidence: {confidence:.2f})")
        
        os.unlink(tfile.name)
