# -*- coding: utf-8 -*-
"""
High-Visibility Quantum Animator
"""
import streamlit as st
import numpy as np
import cv2
import imageio
from PIL import Image
import tempfile
import random
import math

def amplify_quantum_objects(image: np.ndarray) -> list:
    """Enhanced object detection with dynamic thresholds"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if 500 < cv2.contourArea(c) < 100000]

def quantum_object_animation(frame: np.ndarray, progress: float, params: dict) -> np.ndarray:
    """Dynamic object animation with multiple effects"""
    h, w = frame.shape[:2]
    objects = params.get('objects', [])
    
    # Initialize enhanced object states
    if progress == 0:
        detected_objects = amplify_quantum_objects(frame)
        if not detected_objects:
            detected_objects = [(0, 0, w, h)]  # Full frame fallback
        params['objects'] = [{
            'x': x, 'y': y, 
            'dx': random.uniform(-5,5)*params['speed'],
            'dy': random.uniform(-4,4)*params['speed'],
            'w': width, 
            'h': height,
            'angle': 0,
            'd_angle': random.uniform(-4,4),
            'scale': 1.0,
            'd_scale': random.uniform(0.95,1.05)
        } for (x, y, width, height) in detected_objects]

    # Create dynamic background
    bg = frame.copy()
    if params['background_flow']:
        bg = cv2.warpAffine(bg, np.float32([
            [1, 0, 2*params['speed']*math.sin(progress*5)],
            [0, 1, 1*params['speed']*math.cos(progress*3)]
        ]), (w, h))

    # Enhanced object transformations
    for obj in params['objects']:
        obj['x'] = (obj['x'] + obj['dx'] + 2*math.sin(progress*10)) % w
        obj['y'] = (obj['y'] + obj['dy'] + 1.5*math.cos(progress*8)) % h
        obj['angle'] += obj['d_angle']
        obj['scale'] *= obj['d_scale']
        
        # Rotate and scale object
        M = cv2.getRotationMatrix2D((obj['w']/2, obj['h']/2), obj['angle'], obj['scale'])
        obj_img = cv2.warpAffine(
            frame[int(obj['y']):int(obj['y']+obj['h']), 
                 int(obj['x']):int(obj['x']+obj['w'])],
            M, (obj['w'], obj['h'])
        )
        
        # Dynamic blending
        if obj_img.size > 0:
            alpha = 0.7 + 0.3*math.sin(progress*6)
            bg = overlay_transparent(bg, obj_img, int(obj['x']), int(obj['y']), alpha)

    return bg

def overlay_transparent(bg: np.ndarray, fg: np.ndarray, x: int, y: int, alpha: float) -> np.ndarray:
    """Advanced blending with boundary checks"""
    h, w = fg.shape[:2]
    y1, y2 = max(0, y), min(bg.shape[0], y + h)
    x1, x2 = max(0, x), min(bg.shape[1], x + w)
    
    if y2 - y1 <=0 or x2 - x1 <=0: return bg
    
    fg_roi = fg[0:y2-y1, 0:x2-x1]
    bg_roi = bg[y1:y2, x1:x2]
    
    blend = cv2.addWeighted(bg_roi, 1-alpha, fg_roi, alpha, 0)
    bg[y1:y2, x1:x2] = blend
    return bg

def generate_pronounced_frames(image: np.ndarray, params: dict) -> list:
    """Generate high-visibility animation frames"""
    frames = []
    state_params = {
        'speed': params['speed'],
        'background_flow': params['background_flow']
    }
    
    for i in range(params['total_frames']):
        frame = image.copy()
        progress = i / params['total_frames']
        
        frame = quantum_object_animation(frame, progress, state_params)
        frames.append(frame)
        
        # Progressively increase effects
        state_params['speed'] *= 1.005  # Accelerate movement
        
    return frames

def main():
    st.set_page_config(page_title="Quantum FX Animator", layout="centered")
    st.title("🌀 Quantum Amplified Animation")
    
    with st.sidebar:
        st.header("Amplification Controls")
        duration = st.slider("Duration (s)", 2, 10, 5)
        speed = st.slider("Motion Intensity", 1.0, 5.0, 2.5, 0.5)
        effects = st.multiselect("Effects",
            ["Object Animation", "Background Flow"],
            default=["Object Animation", "Background Flow"]
        )
        
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_image = np.array(pil_image)
        processed_image = cv2.resize(orig_image, (768, 768))  # Higher resolution
        
        if st.button("Generate Amplified Animation"):
            params = {
                'total_frames': duration * 24,
                'speed': speed,
                'object_movement': "Object Animation" in effects,
                'background_flow': "Background Flow" in effects
            }
            
            with st.spinner("Generating quantum spectacle..."):
                frames = generate_pronounced_frames(processed_image, params)
                with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
                    imageio.mimsave(tmpfile.name, frames, fps=24, codec='libx264')
                    st.video(tmpfile.name)
            
            st.success("⚡ Electrifying animation complete!")

if __name__ == "__main__":
    main()
