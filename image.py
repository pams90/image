# -*- coding: utf-8 -*-
"""
Quantum Image Animator with Aspect Ratio Lock
"""
import streamlit as st
import numpy as np
import cv2
import imageio
from PIL import Image
import tempfile

def maintain_aspect_ratio(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Quantum-optimized aspect ratio preservation"""
    h, w = image.shape[:2]
    scale = min(max_size/w, max_size/h)
    return cv2.resize(image, (int(w*scale), int(h*scale))) if scale < 1 else image

def quantum_zoom_effect(frame: np.ndarray, progress: float, speed: float) -> np.ndarray:
    """Aspect-preserving zoom transformation"""
    h, w = frame.shape[:2]
    zoom = 1 + progress * speed
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def generate_frames(image: np.ndarray, params: dict) -> list:
    """Quantum frame generator with ratio locking"""
    frames = []
    h, w = image.shape[:2]
    
    for i in range(params['total_frames']):
        frame = image.copy()
        progress = i / params['total_frames']
        
        if params['zoom_effect']:
            frame = quantum_zoom_effect(frame, progress, params['speed'])
            
        if params['pan_effect']:
            pan_step = int((i * params['speed'] * 50) % w)
            frame = np.roll(frame, pan_step, axis=1)
        
        if params['wave_effect']:
            y_coords = np.arange(h)[:, np.newaxis]
            wave_offset = (np.sin(y_coords/h * 4*np.pi + progress*10) * 20 * params['speed']).astype(np.int32)
            frame = np.roll(frame, wave_offset, axis=1)
        
        frames.append(frame)
    
    return frames

def main():
    st.set_page_config(page_title="Willow Aspect-Preserving Animator", layout="centered")
    st.title("🌀 Quantum Image Animator")
    
    with st.sidebar:
        st.header("Animation Controls")
        duration = st.slider("Duration (seconds)", 2, 10, 5)
        speed = st.slider("Effect Intensity", 0.5, 3.0, 1.0)
        effects = st.multiselect("Effects", ["Zoom", "Pan", "Wave"], default=["Zoom"])
    
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_image = np.array(pil_image)
        processed_image = maintain_aspect_ratio(orig_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_image, caption="Original Image")
        with col2:
            st.image(processed_image, caption="Processed Size")
        
        if st.button("Generate Quantum Animation"):
            params = {
                'total_frames': duration * 24,
                'speed': speed,
                'zoom_effect': "Zoom" in effects,
                'pan_effect': "Pan" in effects,
                'wave_effect': "Wave" in effects
            }
            
            with st.spinner("Rendering quantum states..."):
                frames = generate_frames(processed_image, params)
                with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
                    imageio.mimsave(tmpfile.name, frames, fps=24, codec='libx264')
                    st.video(tmpfile.read())
            
            st.success("Quantum animation complete! Original aspect ratio maintained.")

if __name__ == "__main__":
    main()
