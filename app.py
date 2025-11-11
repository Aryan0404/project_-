import streamlit as st
import os
import tempfile
import sys # <-- Import sys

# --- Path Setup ---
# Get the absolute path of the directory containing this script (app.py)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Add this project root to the system path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) # insert at the beginning

# --- Import from your 'utils' package ---
# This will now work because PROJECT_ROOT is in sys.path.
# Python can find 'utils' and 'yes_yes_integration'
try:
    from utils import inference
except ImportError as e:
    st.error(f"Error: Could not import from 'utils' (Error: {e})")
    st.error(f"This error *might* be inside 'utils/inference.py' or 'models/video_resnet.py'.")
    st.error(f"Project Root added to path: {PROJECT_ROOT}")
    st.stop()
except Exception as e:
    st.error(f"A different error occurred on import: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Video Classification", layout="wide")
st.title("📹 VideoResNet Classification App")

# --- Class Names ---
# !!! IMPORTANT: Update this list to match your model's 5 classes
CLASS_NAMES = [
    'Class 1', 
    'Class 2', 
    'Class 3', 
    'Class 4', 
    'Class 5'
]

# Check if the class count matches the model's hardcoded value
if len(CLASS_NAMES) != 5:
    st.warning(
        f"Warning: Your `utils/inference.py` is hardcoded for 5 classes, "
        f"but {len(CLASS_NAMES)} are defined in this app. Please update CLASS_NAMES."
    )

# --- File Uploaders ---
with st.sidebar:
    st.header("1. Upload Model")
    model_file = st.file_uploader("Upload Model Weights (.pth or .pt)", type=["pth", "pt"])
    
    st.header("2. Upload Video")
    video_file = st.file_uploader("Upload Video File (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])

# --- Main Inference Panel ---
if not model_file:
    st.info("Please upload a model file in the sidebar to begin.")
elif not video_file:
    st.info("Please upload a video file to classify.")
else:
    # Use temporary files to securely handle uploads
    model_path = ""
    video_path = ""
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.name)[1]) as t_model:
            t_model.write(model_file.read())
            model_path = t_model.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as t_video:
            t_video.write(video_file.read())
            video_path = t_video.name

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Video")
            st.video(video_path)

        with col2:
            st.subheader("Prediction")
            with st.spinner("Loading model and processing video..."):
                
                # 1. Load Model (from utils.inference)
                model = inference.load_model(model_path)
                
                # 2. Get Prediction (from utils.inference)
                class_name, confidence = inference.predict_video(model, video_path, CLASS_NAMES)

                # --- Display Results ---
                st.success(f"**Predicted Class: {class_name}**")
                st.info(f"**Confidence:** {confidence*100:.2f}%")
                st.progress(confidence)

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")
        st.error("Please ensure your model file is valid and the video is not corrupted.")

    finally:
        # Clean up temporary files
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)