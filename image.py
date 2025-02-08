import streamlit as st
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
import tempfile
from PIL import Image
from io import BytesIO

# Load AI models
@st.cache_resource
def load_models():
    # Load Segment Anything Model (SAM)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load AnimateDiff with motion adapter
    adapter = MotionAdapter.from_pretrained("diffusers/motion-adapter-sd1.5-2-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        motion_adapter=adapter
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler_config)
    pipe.enable_model_cpu_offload()

    return sam, pipe

sam, animate_pipe = load_models()

# Streamlit UI
st.title("🖼️ AI-Powered Image Animation 🎥")
st.sidebar.header("Settings")
motion_strength = st.sidebar.slider("Motion Strength", 1.0, 2.0, 1.2)
video_duration = st.sidebar.slider("Video Duration (seconds)", 2, 5, 3)

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read and process image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert to numpy array for SAM
    img_np = np.array(image)
    
    # Object selection
    st.markdown("## Select Object to Animate")
    st.write("Click on the object you want to animate")
    
    # Get click coordinates
    click_container = st.empty()
    with click_container:
        click_coords = st.data_editor(
            [{"x": 0, "y": 0}],
            column_config={
                "x": st.column_config.NumberColumn("X coordinate"),
                "y": st.column_config.NumberColumn("Y coordinate")
            },
            hide_index=True,
            key="coords"
        )
    
    if st.button("Segment Object"):
        # Run SAM with point input
        predictor = SamPredictor(sam)
        predictor.set_image(img_np)
        
        input_point = np.array([[click_coords[0]["x"], click_coords[0]["y"]]])
        input_label = np.array([1])  # Positive label
        
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Select best mask
        best_mask = masks[np.argmax(scores)].astype(np.uint8)
        
        # Apply mask to original image
        masked_image = cv2.bitwise_and(img_np, img_np, mask=best_mask)
        st.image(masked_image, caption="Segmented Object", use_column_width=True)
        
        # Generate animation
        with st.spinner("Generating animation..."):
            # Convert to PIL Image
            input_image = Image.fromarray(masked_image).resize((512, 512))
            
            # Generate animation frames
            output = animate_pipe(
                image=input_image,
                prompt="professional high-quality animated movie, smooth motion",  # Generic motion prompt
                guidance_scale=motion_strength,
                num_inference_steps=25,
                num_frames=video_duration * 8,  # 8fps for shorter generation
            )
            
            # Save to GIF
            gif_bytes = export_to_gif(output.frames[0], "animation.gif")
            
            # Display and download
            st.video(gif_bytes, format="video/mp4")
            st.download_button(
                label="Download Animation",
                data=gif_bytes,
                file_name="animated_object.mp4",
                mime="video/mp4"
            )
