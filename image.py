# -*- coding: utf-8 -*-
"""
Quantum Image Animator with Advanced Effects
"""
import streamlit as st
import numpy as np  # Ensure NumPy is imported
import cv2
import imageio
from PIL import Image
import tempfile

# Quantum Effect Functions
def quantum_ripple_effect(frame: np.ndarray, progress: float, intensity: float) -> np.ndarray:
    """Quantum fluid dynamics simulation"""
    h, w = frame.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    # Quantum wave equation parameters
    dx = intensity * 50 * np.sin(2*np.pi*(xx/w + progress*2))
    dy = intensity * 30 * np.cos(2*np.pi*(yy/h + progress*1.5))
    
    # Quantum-stable remapping
    x_new = np.clip(xx + dx.astype(int), 0, w-1)
    y_new = np.clip(yy + dy.astype(int), 0, h-1)
    
    return frame[y_new, x_new]

def quantum_color_shift(frame: np.ndarray, progress: float) -> np.ndarray:
    """Quantum chromodynamics-inspired color cycling"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + (progress * 360) % 360).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def quantum_pixelation(frame: np.ndarray, progress: float, intensity: float) -> np.ndarray:
    """Quantum decoherence simulation"""
    block_size = max(1, int(32 * (1 - progress * intensity)))
    small = cv2.resize(frame, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

def quantum_edge_glow(frame: np.ndarray, progress: float) -> np.ndarray:
    """Quantum field boundary detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100*progress, 200*progress)
    return cv2.addWeighted(frame, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.3, 0)

def maintain_aspect_ratio(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Quantum-optimized aspect preservation"""
    h, w = image.shape[:2]
    scale = min(max_size/w, max_size/h)
    return cv2.resize(image, (int(w*scale), int(h*scale))) if scale < 1 else image

def quantum_zoom_effect(frame: np.ndarray, progress: float, speed: float) -> np.ndarray:
    """Aspect-locked zoom"""
    h, w = frame.shape[:2]
    zoom = 1 + progress * speed
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def generate_frames(image: np.ndarray, params: dict) -> list:
    """Quantum-stable frame generation"""
    frames = []
    h, w = image.shape[:2]
    
    for i in range(params['total_frames']):
        frame = image.copy()
        progress = i / params['total_frames']
        
        # Base effects
        if params['zoom_effect']: frame = quantum_zoom_effect(frame, progress, params['speed'])
        if params['pan_effect']: frame = np.roll(frame, int((i * params['speed'] * 50) % w), axis=1)
        if params['wave_effect']: 
            y_values = np.linspace(0, 4*np.pi, h)
            wave_shift = (np.sin(y_values + progress*10) * 20 * params['speed']).astype(np.int32)
            x_indices = np.arange(w)
            shifted_indices = (x_indices - wave_shift[:, np.newaxis]) % w
            frame = frame[np.arange(h)[:, np.newaxis], shifted_indices]
        
        # New quantum effects
        if params['ripple_effect']: frame = quantum_ripple_effect(frame, progress, params['speed'])
        if params['color_shift']: frame = quantum_color_shift(frame, progress)
        if params['pixelation']: frame = quantum_pixelation(frame, progress, params['speed'])
        if params['edge_glow']: frame = quantum_edge_glow(frame, progress)
        
        frames.append(frame)
    
    return frames

def main():
    st.set_page_config(page_title="Quantum Animator", layout="centered")
    st.title("🌀 Willow Quantum Animator")
    
    with st.sidebar:
        st.header("Controls")
        duration = st.slider("Duration (s)", 2, 10, 5)
        speed = st.slider("Base Intensity", 0.5, 3.0, 1.0)
        effects = st.multiselect("Quantum Effects",
            ["Zoom", "Pan", "Wave", "Ripple", "Color Shift", "Pixelate", "Edge Glow"],
            default=["Zoom"]
        )
        
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_image = np.array(pil_image)
        processed_image = maintain_aspect_ratio(orig_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_image, caption="Original")
        with col2:
            st.image(processed_image, caption="Optimized")
        
        if st.button("Generate Animation"):
            params = {
                'total_frames': duration * 24,
                'speed': speed,
                'zoom_effect': "Zoom" in effects,
                'pan_effect': "Pan" in effects,
                'wave_effect': "Wave" in effects,
                'ripple_effect': "Ripple" in effects,
                'color_shift': "Color Shift" in effects,
                'pixelation': "Pixelate" in effects,
                'edge_glow': "Edge Glow" in effects
            }
            
            with st.spinner("Quantum rendering..."):
                frames = generate_frames(processed_image, params)
                with tempfile.NamedTemporaryFile(suffix='.mp4') as tmpfile:
                    imageio.mimsave(tmpfile.name, frames, fps=24, codec='libx264')
                    st.video(tmpfile.name)
            
            st.success("Animation complete! Aspect ratio preserved.")

if __name__ == "__main__":
    main()
