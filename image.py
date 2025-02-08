import streamlit as st
import numpy as np
import cv2
import moviepy.editor as mp
from tempfile import NamedTemporaryFile

# Streamlit UI
st.title("🖼️ Image Animation with Motion! 🎬")
st.sidebar.header("Settings")
motion_speed = st.sidebar.slider("Motion Speed", 1, 10, 5)
video_duration = st.sidebar.slider("Video Duration (seconds)", 2, 10, 5)

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Get image dimensions
    height, width, _ = img.shape

    # Detect objects using OpenCV (simple contour detection)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create layers for detected objects
    object_layers = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        object_layers.append((img[y:y+h, x:x+w], (x, y, w, h)))

    # Create frames with motion
    frames = []
    for t in range(video_duration * 24):  # Assuming 24 FPS
        frame = img.copy()
        for obj, (x, y, w, h) in object_layers:
            # Move objects randomly
            dx = int(np.sin(t / 10) * motion_speed)
            dy = int(np.cos(t / 10) * motion_speed)
            new_x, new_y = x + dx, y + dy

            # Ensure objects stay within bounds
            new_x = max(0, min(new_x, width - w))
            new_y = max(0, min(new_y, height - h))

            frame[new_y:new_y+h, new_x:new_x+w] = obj  # Place object at new position

        frames.append(frame)

    # Convert frames to a video
    clip = mp.ImageSequenceClip(frames, fps=24)
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    clip.write_videofile(temp_file.name, codec="libx264")

    # Display Video
    st.video(temp_file.name)

    # Provide Download Option
    with open(temp_file.name, "rb") as file:
        st.download_button("📥 Download Animated Video", file, file_name="animated_motion.mp4", mime="video/mp4")
