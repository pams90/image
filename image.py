import streamlit as st
import numpy as np
import cv2
import os
import moviepy.editor as mp
from moviepy.video.fx import fadein, fadeout
from moviepy.video.fx import resize
from moviepy.video.fx import scroll
from tempfile import NamedTemporaryFile

# Title
st.title("📷 Image to Animated Video Converter 🎬")

# Sidebar options
st.sidebar.header("Settings")
effect = st.sidebar.selectbox("Choose an animation effect:", ["Pan", "Zoom", "Fade"])
duration = st.sidebar.slider("Video Duration (seconds)", 2, 10, 5)
speed = st.sidebar.slider("Animation Speed", 0.5, 3.0, 1.0)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert OpenCV BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Get image dimensions
    height, width, _ = img.shape

    # Define output video parameters
    fps = 24
    output_path = "output.mp4"

    def apply_pan_effect():
        """Apply a pan effect (left to right scrolling)."""
        clip = mp.ImageClip(img).set_duration(duration)
        clip = scroll.scroll(clip, w=width, x_speed=int(speed * 10))
        return clip

    def apply_zoom_effect():
        """Apply a zoom-in effect."""
        clip = mp.ImageClip(img).set_duration(duration)
        clip = resize.resize(clip, lambda t: 1 + 0.2 * (t / duration))
        return clip

    def apply_fade_effect():
        """Apply a fade-in and fade-out effect."""
        clip = mp.ImageClip(img).set_duration(duration)
        clip = fadein.fadein(clip, 1).fx(fadeout.fadeout, 1)
        return clip

    # Select animation effect
    if effect == "Pan":
        final_clip = apply_pan_effect()
    elif effect == "Zoom":
        final_clip = apply_zoom_effect()
    elif effect == "Fade":
        final_clip = apply_fade_effect()

    # Save video
    final_clip.write_videofile(output_path, fps=fps, codec="libx264", audio=False)

    # Display the video
    st.video(output_path)

    # Provide download link
    with open(output_path, "rb") as file:
        st.download_button("📥 Download Video", file, file_name="animated_video.mp4", mime="video/mp4")

