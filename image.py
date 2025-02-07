import streamlit as st
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
from PIL import Image
import os
import requests

# URL of the checkpoint file
CHECKPOINT_URL = "https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

# Function to download the checkpoint file
def download_checkpoint(url, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        st.info("Downloading checkpoint file...")
        response = requests.get(url)
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)
        st.success("Checkpoint file downloaded.")

# Download the checkpoint file if it doesn't exist
download_checkpoint(CHECKPOINT_URL, CHECKPOINT_PATH)

# Load AI models
@st.experimental_singleton
def load_models():
    # Load Segment Anything Model (SAM)
    sam_checkpoint = CHECKPOINT_PATH
    if not os.path.isfile(sam_checkpoint):
        st.error(f"Checkpoint file '{sam_checkpoint}' not found.")
        return None, None
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    with open(sam_checkpoint, "rb") as f:
        state_dict = torch.load(f, weights_only=False)
    sam.load_state_dict(state_dict)
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

if sam is None or animate_pipe is None:
    st.stop()

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

    # Object selection via Streamlit's interactive features
    st.markdown("## Select Object to Animate")
    click_coords = st.slider(
        "Click the object to animate by specifying coordinates:",
        min_value=0, max_value=image.size[0], value=(0, 0),
        step=1, format="%(value)s"
    )

    if st.button("Segment Object"):
        # Run SAM with point input
        predictor = SamPredictor(sam)
        predictor.set_image(img_np)

        input_point = np.array([[click_coords[0], click_coords[1]]])
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

            # Save to GIF (for video conversion)
            gif_bytes = export_to_gif(output.frames[0], "animation.gif")

            # Convert gif to MP4
            video_bytes = gif_bytes  # Placeholder for future gif-to-mp4 conversion logic

            # Display and download
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                label="Download Animation",
                data=video_bytes,
                file_name="animated_object.mp4",
                mime="video/mp4"
            )
else:
    st.warning("Please upload an image to proceed.")
