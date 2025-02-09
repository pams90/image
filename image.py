# -*- coding: utf-8 -*-
"""
Quantum Animator with Vidnoz-Like Effects
"""
import streamlit as st
import numpy as np
import cv2
import imageio
from PIL import Image
import tempfile
import random
import math
from moviepy.editor import VideoFileClip, AudioFileClip

# Enhanced Effects
def parallax_effect(frame: np.ndarray, progress: float, intensity: float) -> np.ndarray:
    """Simulate depth by moving foreground and background at different speeds"""
    h, w = frame.shape[:2]
    bg = cv2.warpAffine(frame, np.float32([[1,0,0.5*progress*intensity],[0,1,0.2*progress*intensity]]), (w,h))
    fg = cv2.warpAffine(frame, np.float32([[1,0,2*progress*intensity],[0,1,1*progress*intensity]]), (w,h))
    return cv2.addWeighted(bg, 0.7, fg, 0.3, 0)

def natural_motion(frame: np.ndarray, progress: float, intensity: float) -> np.ndarray:
    """Simulate natural movements like swaying or flowing"""
    h, w = frame.shape[:2]
    wave_matrix = np.zeros_like(frame)
    for y in range(h):
        wave_matrix[y,:] = int(20 * intensity * np.sin(2*np.pi*y/h + progress*5))
    return cv2.addWeighted(frame, 0.8, wave_matrix, 0.2, 0)

def add_text(frame: np.ndarray, text: str, position: tuple, font_scale: float = 1.0) -> np.ndarray:
    """Add dynamic text overlays"""
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255,255,255), 2, cv2.LINE_AA)

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
        
        # Apply effects
        if params['parallax']:
            frame = parallax_effect(frame, progress, params['speed'])
        if params['natural_motion']:
            frame = natural_motion(frame, progress, params['speed'])
        if params['object_movement']:
            frame = quantum_object_animation(frame, progress, state_params)
        if params['text_overlay']:
            frame = add_text(frame, "Quantum Animator", (50, 50), font_scale=1.5)
        
        frames.append(frame)
        
        # Progressively increase effects
        state_params['speed'] *= 1.005  # Accelerate movement
        
    return frames

def add_music(video_path: str, audio_path: str, output_path: str) -> None:
    """Add background music to video"""
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(output_path, codec='libx264')

def main():
    st.set_page_config(page_title="Quantum FX Animator", layout="centered")
    st.title("🌀 Quantum Amplified Animation")
    
    with st.sidebar:
        st.header("Amplification Controls")
        duration = st.slider("Duration (s)", 2, 10, 5)
        speed = st.slider("Motion Intensity", 1.0, 5.0, 2.5, 0.5)
        effects = st.multiselect("Effects",
            ["Object Animation", "Background Flow", "Parallax", "Natural Motion", "Text Overlay"],
            default=["Object Animation", "Background Flow"]
        )
        
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    audio_file = st.file_uploader("Upload Background Music (optional)", type=["mp3", "wav"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_image = np.array(pil_image)
        processed_image = cv2.resize(orig_image, (768, 768))  # Higher resolution
        
        if st.button("Generate Amplified Animation"):
            params = {
                'total_frames': duration * 24,
                'speed': speed,
                'object_movement': "Object Animation" in effects,
                'background_flow': "Background Flow" in effects,
                'parallax': "Parallax" in effects,
                'natural_motion': "Natural Motion" in effects,
                'text_overlay': "Text Overlay" in effects
            }
            
            with st.spinner("Generating quantum spectacle..."):
                frames = generate_pronounced_frames(processed_image, params)
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmpfile:
                    video_path = tmpfile.name
                    imageio.mimsave(video_path, frames, fps=24, codec='libx264')
                    
                    if audio_file:
                        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as audio_tmp:
                            audio_tmp.write(audio_file.read())
                            audio_path = audio_tmp.name
                            output_path = video_path.replace(".mp4", "_with_audio.mp4")
                            add_music(video_path, audio_path, output_path)
                            st.video(output_path)
                    else:
                        st.video(video_path)
            
            st.success("⚡ Electrifying animation complete!")

if __name__ == "__main__":
    main()
