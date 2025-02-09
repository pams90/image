# -*- coding: utf-8 -*-
"""
Quantum Object Animator with Debugging
"""
import streamlit as st
import numpy as np
import cv2
import imageio
from PIL import Image
import tempfile
import random

# Debugging function
def debug_frame(frame: np.ndarray, name: str = "frame") -> None:
    """Display frame statistics for debugging"""
    st.sidebar.write(f"🔍 {name} debug:")
    st.sidebar.write(f"- Shape: {frame.shape}")
    st.sidebar.write(f"- Min/Max: {frame.min()}, {frame.max()}")
    st.sidebar.write(f"- Mean: {frame.mean():.2f}")

# Quantum Object Movement Functions
def detect_quantum_objects(image: np.ndarray) -> list:
    """Quantum-inspired object detection using edge analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]

def quantum_object_movement(frame: np.ndarray, progress: float, params: dict) -> np.ndarray:
    """Move detected objects with individual quantum trajectories"""
    h, w = frame.shape[:2]
    objects = params.get('objects', [])
    
    # Initialize object states on first frame
    if progress == 0:
        detected_objects = detect_quantum_objects(frame)
        if not detected_objects:
            st.warning("⚠️ No objects detected! Using default movement.")
            detected_objects = [(50, 50, 100, 100)]  # Fallback object
        params['objects'] = [{
            'x': x, 'y': y, 
            'dx': random.uniform(-3, 3)*params['speed'], 
            'dy': random.uniform(-2, 2)*params['speed'],
            'w': width, 
            'h': height
        } for (x, y, width, height) in detected_objects]

    # Create background with subtle motion
    bg = frame.copy()
    if params['background_flow']:
        bg = cv2.warpAffine(bg, np.float32([[1, 0, 0.5*params['speed']], [0, 1, 0.3*params['speed']]]), (w, h))

    # Move each object
    for obj in params['objects']:
        # Update positions with quantum uncertainty
        obj['x'] = (obj['x'] + obj['dx'] + random.uniform(-0.5, 0.5)) % w
        obj['y'] = (obj['y'] + obj['dy'] + random.uniform(-0.3, 0.3)) % h
        
        # Extract object and blend with background
        obj_img = frame[int(obj['y']):int(obj['y']+obj['h']), 
                        int(obj['x']):int(obj['x']+obj['w'])]
        if obj_img.size > 0:
            bg[int(obj['y']):int(obj['y']+obj['h']), 
               int(obj['x']):int(obj['x']+obj['w'])] = cv2.addWeighted(
                bg[int(obj['y']):int(obj['y']+obj['h']), 
                   int(obj['x']):int(obj['x']+obj['w'])],
                0.3,
                obj_img,
                0.7,
                0
            )

    return bg

# Frame generator
def generate_frames(image: np.ndarray, params: dict) -> list:
    """Generate frames with moving objects"""
    frames = []
    h, w = image.shape[:2]
    state_params = {'speed': params['speed'], 'background_flow': params['background_flow']}
    
    for i in range(params['total_frames']):
        frame = image.copy()
        progress = i / params['total_frames']
        
        # Apply object movement
        if params['object_movement']:
            frame = quantum_object_movement(frame, progress, state_params)
            
        frames.append(frame)
    
    return frames

# Main app
def main():
    st.set_page_config(page_title="Quantum Object Animator", layout="centered")
    st.title("🌀 Quantum Object Movement")
    
    with st.sidebar:
        st.header("Controls")
        duration = st.slider("Duration (s)", 2, 10, 5)
        speed = st.slider("Movement Speed", 0.5, 3.0, 1.0)
        effects = st.multiselect("Effects",
            ["Object Movement", "Background Flow"],
            default=["Object Movement"]
        )
        debug = st.checkbox("Enable Debugging")
        
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            orig_image = np.array(pil_image)
            
            if debug:
                debug_frame(orig_image, "Original Image")
            
            processed_image = cv2.resize(orig_image, (512, 512))  # Fixed size for stability
            
            if st.button("Generate Animation"):
                params = {
                    'total_frames': duration * 24,
                    'speed': speed,
                    'object_movement': "Object Movement" in effects,
                    'background_flow': "Background Flow" in effects
                }
                
                with st.spinner("Rendering quantum animation..."):
                    frames = generate_frames(processed_image, params)
                    if debug:
                        debug_frame(frames[0], "First Frame")
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
                        imageio.mimsave(tmpfile.name, frames, fps=24, codec='libx264')
                        st.video(tmpfile.name)
                
                st.success("🎉 Animation complete!")
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
            if debug:
                st.exception(e)

if __name__ == "__main__":
    main()
