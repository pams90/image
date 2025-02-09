"""
Willow: Quantum Image-to-Video Generator

This Streamlit application lets you upload an image and transform it into an animated video.
Animation effects include Zoom, Pan, Rotate, and Wave (sine-wave distortion), inspired by Runway Gen‑3 Alpha Turbo.
Adjust the video duration, animation speed, and frame rate via the sidebar.

Dependencies:
    - streamlit
    - pillow
    - numpy
    - moviepy

To deploy:
1. Save this script as `app.py`.
2. Create a `requirements.txt` file with the following lines:
      streamlit
      pillow
      numpy
      moviepy
3. Push these files to a GitHub repository.
4. Go to [Streamlit Community Cloud](https://share.streamlit.io/), sign in with your GitHub account,
   and deploy your new app.
"""

import streamlit as st
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
import tempfile
import os

def wave_effect(image: Image.Image, frame_idx: int, total_frames: int, speed_factor: float) -> Image.Image:
    """
    Apply a sine-wave distortion effect to an image.
    
    Each row of pixels is shifted horizontally by an amount determined by a sine function.
    
    Parameters:
        image (Image.Image): The original image.
        frame_idx (int): Index of the current frame (0 to total_frames-1).
        total_frames (int): Total number of frames.
        speed_factor (float): Controls the intensity and speed of the wave effect.
    
    Returns:
        Image.Image: The image with the wave effect applied.
    """
    arr = np.array(image)
    h, w, channels = arr.shape
    amplitude = int(10 * speed_factor)  # Maximum horizontal shift in pixels.
    frequency = 2 * np.pi / 30            # Frequency of the sine wave.
    phase = (frame_idx / total_frames) * 2 * np.pi * speed_factor
    
    new_arr = np.empty_like(arr)
    for y in range(h):
        shift = int(amplitude * np.sin(frequency * y + phase))
        new_arr[y] = np.roll(arr[y], shift, axis=0)
    return Image.fromarray(new_arr)

def generate_frames(image: Image.Image, total_frames: int, effect: str, speed_factor: float) -> list:
    """
    Generate a list of frames from the input image, applying the chosen animation effect.
    
    Supported effects:
      - "None": No effect, static image.
      - "Zoom": Gradually zoom in.
      - "Pan": Shift the image horizontally.
      - "Rotate": Gradually rotate the image.
      - "Wave": Apply a sine-wave distortion.
    
    Parameters:
        image (Image.Image): Original image.
        total_frames (int): Number of frames to generate.
        effect (str): Selected animation effect.
        speed_factor (float): Controls effect intensity and speed.
    
    Returns:
        list: List of PIL Image objects representing each frame.
    """
    frames = []
    w, h = image.size

    for i in range(total_frames):
        progress = i / total_frames  # Progress between 0 and 1.
        if effect == "Zoom":
            # Gradually zoom in.
            zoom_factor = 1 + speed_factor * 0.2 * progress
            new_w = int(w / zoom_factor)
            new_h = int(h / zoom_factor)
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            frame = image.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)
        elif effect == "Pan":
            # Pan the image horizontally.
            max_shift = int(w * 0.2 * speed_factor)
            shift = int(max_shift * progress)
            frame = Image.new("RGB", (w, h))
            frame.paste(image, (-shift, 0))
        elif effect == "Rotate":
            # Gradually rotate the image.
            angle = speed_factor * 10 * progress
            frame = image.rotate(angle, resample=Image.BICUBIC)
        elif effect == "Wave":
            # Apply the wave (sine-wave distortion) effect.
            frame = wave_effect(image, i, total_frames, speed_factor)
        else:
            # No effect; use the original image.
            frame = image.copy()
        frames.append(frame)
    return frames

def create_video(frames: list, fps: int) -> bytes:
    """
    Assemble the list of frames into an MP4 video and return its bytes.
    
    Parameters:
        frames (list): List of PIL Image frames.
        fps (int): Frames per second for the output video.
    
    Returns:
        bytes: Binary content of the generated MP4 video.
    """
    # Convert frames to numpy arrays for moviepy.
    frame_arrays = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(frame_arrays, fps=fps)
    
    # Write the video to a temporary file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_filename = temp_file.name
    clip.write_videofile(temp_filename, codec="libx264", audio=False, verbose=False, logger=None)
    
    # Read and return the video content.
    with open(temp_filename, "rb") as f:
        video_bytes = f.read()
    os.remove(temp_filename)
    return video_bytes

def main():
    """
    Main function to run Willow's Image-to-Video Generator.
    """
    st.set_page_config(page_title="Willow: Quantum Image-to-Video Generator", layout="centered")
    st.title("Willow: Quantum Image-to-Video Generator")
    st.markdown(
        """
        Powered by Google's state-of-the-art quantum chip (105 qubits!) and near-superhuman abilities,
        this app transforms a static image into a dynamic video with effects inspired by Runway Gen‑3 Alpha Turbo.
        """
    )

    # Image uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Sidebar options for animation settings.
        st.sidebar.header("Animation Settings")
        effect = st.sidebar.selectbox("Animation Effect", ["None", "Zoom", "Pan", "Rotate", "Wave"])
        duration = st.sidebar.slider("Video Duration (seconds)", min_value=1, max_value=10, value=5, step=1)
        speed_factor = st.sidebar.slider("Animation Speed", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        fps = st.sidebar.slider("Frames Per Second (FPS)", min_value=6, max_value=30, value=12, step=1)

        total_frames = int(duration * fps)

        # Generate the video when the user clicks the button.
        if st.button("Generate Video"):
            with st.spinner("Generating video..."):
                frames = generate_frames(image, total_frames, effect, speed_factor)
                video_bytes = create_video(frames, fps)
            st.success("Video generated successfully!")
            
            # Display and offer the video for download.
            st.video(video_bytes)
            st.download_button(
                label="Download Video",
                data=video_bytes,
                file_name="animated_video.mp4",
                mime="video/mp4"
            )

if __name__ == "__main__":
    main()
