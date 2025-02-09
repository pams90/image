import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imageio
import tempfile
import os

# Function to apply animation effects
def apply_animation_effect(image, effect, frame_count, speed):
    frames = []
    height, width, _ = image.shape
    
    for i in range(frame_count):
        frame = image.copy()
        
        if effect == "zoom":
            scale = 1 + (i / frame_count) * speed
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
            frame = frame[int((frame.shape[0] - height) / 2):int((frame.shape[0] + height) / 2),
                          int((frame.shape[1] - width) / 2):int((frame.shape[1] + width) / 2)]
        
        elif effect == "rotate":
            angle = i * speed
            M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
            frame = cv2.warpAffine(frame, M, (width, height))
        
        elif effect == "pan":
            shift_x = int((i / frame_count) * speed * width)
            frame = np.roll(frame, shift_x, axis=1)
        
        frames.append(frame)
    
    return frames

# Function to create video from frames
def create_video(frames, duration, output_path):
    frame_rate = len(frames) / duration
    with imageio.get_writer(output_path, fps=frame_rate) as writer:
        for frame in frames:
            writer.append_data(frame)

# Streamlit app
def main():
    st.title("Image to Animated Video Converter")
    st.write("Upload an image and customize the animation to create a video.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Animation options
        st.sidebar.header("Animation Settings")
        effect = st.sidebar.selectbox("Select Animation Effect", ["zoom", "rotate", "pan"])
        duration = st.sidebar.slider("Video Duration (seconds)", 1, 10, 5)
        speed = st.sidebar.slider("Animation Speed", 0.1, 2.0, 1.0)
        frame_count = st.sidebar.slider("Number of Frames", 10, 100, 50)
        
        # Apply animation
        frames = apply_animation_effect(image, effect, frame_count, speed)
        
        # Create video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            output_path = tmpfile.name
            create_video(frames, duration, output_path)
            
            # Display the video
            st.video(output_path)
            
            # Download link
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download Video",
                    data=file,
                    file_name="animated_video.mp4",
                    mime="video/mp4"
                )
        
        # Clean up
        os.unlink(output_path)

if __name__ == "__main__":
    main()
