import streamlit as st
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from diffusers import AnimateDiffPipeline
import moviepy.editor as mp
import tempfile

# Load AI models
@st.cache_resource
def load_models():
    # Load Segment Anything Model (SAM)
    sam_checkpoint = "sam_vit_h.pth"  # Download model if needed
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load AnimateDiff (for AI motion)
    animate_diff = AnimateDiffPipeline.from_pretrained("AnimateDiff/sdxl")
    animate_diff.to("cuda" if torch.cuda.is_available() else "cpu")

    return sam, animate_diff

sam, animate_diff = load_models()

# Streamlit UI
st.title("🖼️ AI-Powered Image Animation 🎥")
st.sidebar.header("Settings")
motion_strength = st.sidebar.slider("Motion Strength", 1, 10, 5)
video_duration = st.sidebar.slider("Video Duration (seconds)", 2, 10, 5)

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display Image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # **Step 1: Extract Objects using SAM**
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, _, _ = predictor.predict()  # Get object masks

    # **Step 2: Generate AI Motion using AnimateDiff**
    prompt = "Generate realistic motion for the foreground elements"
    animation_frames = animate_diff(prompt, num_inference_steps=motion_strength, video_length=video_duration * 24)

    # Convert frames to video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    animation_clip = mp.ImageSequenceClip([np.array(f) for f in animation_frames], fps=24)
    animation_clip.write_videofile(temp_video.name, codec="libx264")

    # Display AI-generated Video
    st.video(temp_video.name)

    # Provide Download Option
    with open(temp_video.name, "rb") as file:
        st.download_button("📥 Download AI Animated Video", file, file_name="ai_motion.mp4", mime="video/mp4")
