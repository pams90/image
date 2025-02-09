import streamlit as st
import numpy as np
from PIL import Image
import cv2
import imageio
import tempfile

class QuantumAnimationEngine:
    def __init__(self, image):
        self.base = np.array(image, dtype=np.float32)
        self.h, self.w, _ = self.base.shape
        self.quantum_fields = self._init_quantum_fields()
        
    def _init_quantum_fields(self):
        """Initialize quantum state matrices for all effects"""
        x = np.linspace(0, 10*np.pi, self.w)
        y = np.linspace(0, 10*np.pi, self.h)
        xx, yy = np.meshgrid(x, y)
        
        return {
            'waterfall': np.sin(xx*2 + yy/4),
            'clouds': np.cos(xx/3 + yy*2),
            'aurora': np.stack([xx, yy, xx+yy], axis=-1),
            'stars': np.random.rand(self.h, self.w),
            'sakura': np.sin(xx)*np.cos(yy/2),
            'ripple': np.sqrt((xx - self.w/2)**2 + (yy - self.h/2)**2),
            'wind': np.random.randn(self.h, self.w, 2)
        }

    def generate_effect(self, frame, t, effect, params):
        """Quantum effect dispatcher with temporal coherence"""
        effect_map = {
            "Waterfall Flow": self._waterfall_effect,
            "Cloud Drift": self._cloud_effect,
            "Aurora Borealis": self._aurora_effect,
            "Star Particles": self._star_effect,
            "Sakura Storms": self._sakura_effect,
            "Quantum Ripple": self._ripple_effect,
            "Temporal Zoom": self._zoom_effect,
            "Dynamic Wind": self._wind_effect
        }
        return effect_map[effect](frame, t, params)

    def _waterfall_effect(self, frame, t, params):
        """Vertical flow with dynamic splashing"""
        flow = params['intensity'] * 40 * np.sin(t/2 + self.quantum_fields['waterfall'])
        return cv2.remap(frame,
                        np.indices((self.h, self.w))[1].astype(np.float32),
                        (np.indices((self.h, self.w))[0] + flow).astype(np.float32),
                        cv2.INTER_LINEAR)

    def _cloud_effect(self, frame, t, params):
        """Parallax cloud movement with depth simulation"""
        shift = params['speed'] * 25 * np.cos(t/3 + self.quantum_fields['clouds'])
        return np.roll(frame, int(shift.mean()), axis=1)

    def _aurora_effect(self, frame, t, params):
        """Northern lights simulation with color shifts"""
        hue_shift = params['intensity'] * 30 * np.sin(t/5 + self.quantum_fields['aurora'])
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[...,0] = (hsv[...,0] + hue_shift[...,0]) % 180
        hsv[...,1] = np.clip(hsv[...,1] * (1 + params['speed']/10), 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _star_effect(self, frame, t, params):
        """Twinkling star particles with dynamic intensity"""
        stars = (self.quantum_fields['stars'] > 0.99 - params['intensity']/10) * 255
        return np.clip(frame + stars[...,None] * np.array([255, 255, 200]), 0, 255)

    def _sakura_effect(self, frame, t, params):
        """Petal storm with physics simulation"""
        dx = params['intensity'] * 15 * np.sin(t/3 + self.quantum_fields['sakura'])
        dy = params['speed'] * 8 * np.cos(t/2 + self.quantum_fields['sakura']*2)
        return cv2.remap(frame, 
                        (np.indices((self.h, self.w))[1] + dx).astype(np.float32),
                        (np.indices((self.h, self.w))[0] + dy).astype(np.float32),
                        cv2.INTER_LANCZOS4)

    def _ripple_effect(self, frame, t, params):
        """Quantum interference patterns"""
        distortion = params['intensity'] * 20 * np.sin(t/2 + self.quantum_fields['ripple']/50)
        return cv2.remap(frame,
                        (np.indices((self.h, self.w))[1] + distortion).astype(np.float32),
                        (np.indices((self.h, self.w))[0] + distortion).astype(np.float32),
                        cv2.INTER_CUBIC)

    def _zoom_effect(self, frame, t, params):
        """Temporal-consistent zoom with rotation"""
        scale = 1 + params['intensity'] * 0.5 * np.sin(t/3)
        angle = params['speed'] * 15 * t
        M = cv2.getRotationMatrix2D((self.w/2, self.h/2), angle, scale)
        return cv2.warpAffine(frame, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT)

    def _wind_effect(self, frame, t, params):
        """Dynamic wind simulation with directional control"""
        wind_vec = self.quantum_fields['wind'] * params['intensity'] * 30
        return cv2.remap(frame,
                        (np.indices((self.h, self.w))[1] + wind_vec[...,0]*np.sin(t/3)).astype(np.float32),
                        (np.indices((self.h, self.w))[0] + wind_vec[...,1]*np.cos(t/3)).astype(np.float32),
                        cv2.INTER_LANCZOS4)

def main():
    st.set_page_config(page_title="Quantum Animation Studio", layout="wide")
    
    with st.sidebar:
        st.header("Quantum Controls")
        effect = st.selectbox("Animation Effect", [
            "Waterfall Flow", 
            "Cloud Drift",
            "Aurora Borealis",
            "Star Particles",
            "Sakura Storms",
            "Quantum Ripple",
            "Temporal Zoom",
            "Dynamic Wind"
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("Duration (seconds)", 3.0, 30.0, 10.0)
            fps = st.select_slider("FPS", [24, 30, 60], 30)
        with col2:
            intensity = st.slider("Effect Intensity", 0.1, 3.0, 1.5)
            speed = st.slider("Animation Speed", 0.5, 2.5, 1.0)

    uploaded = st.file_uploader("Upload Base Image", type=["png", "jpg", "jpeg"])
    
    if uploaded:
        with st.spinner("Generating quantum animation..."):
            img = Image.open(uploaded).convert("RGB")
            engine = QuantumAnimationEngine(img)
            
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                frames = []
                for i in range(int(duration * fps)):
                    frame = engine.generate_effect(engine.base, i/fps, effect, {
                        'intensity': intensity,
                        'speed': speed
                    }).astype(np.uint8)
                    frames.append(frame)
                
                imageio.mimwrite(tmp.name, frames, fps=fps, 
                               codec="libx264", output_params=["-crf", "20"])
                
                st.video(tmp.name)
                st.download_button("Download Animation", tmp.name,
                                 file_name="quantum_animation.mp4",
                                 mime="video/mp4")

if __name__ == "__main__":
    main()
