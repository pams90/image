import streamlit as st  
import numpy as np  
from PIL import Image  
import cv2  
import imageio  
import tempfile  

class SakuraAnimationEngine:  
    def __init__(self, image):  
        self.base = np.array(image, dtype=np.float32)  
        self.h, self.w, _ = self.base.shape  
        self.petal_field = self._create_petal_map()  

    def _create_petal_map(self):  
        """Quantum-generated petal distribution map"""  
        x = np.linspace(0, 8*np.pi, self.w)  
        y = np.linspace(0, 8*np.pi, self.h)  
        return np.sin(x[None,:]) * np.cos(y[:,None]) * 0.7  

    def _sakura_effect(self, frame, t):  
        """Combined petal fall and water reflection effects"""  
        # Petal motion  
        petal_dx = 15 * np.sin(t/3 + self.petal_field)  
        petal_dy = 8 * np.cos(t/2 + self.petal_field*2)  
        frame = cv2.remap(frame,  
                        (np.indices((self.h, self.w))[1] + petal_dx).astype(np.float32),  
                        (np.indices((self.h, self.w))[0] + petal_dy).astype(np.float32),  
                        cv2.INTER_LANCZOS4)  

        # Water reflection  
        reflection = frame[::-1,:,:]  
        reflection = cv2.GaussianBlur(reflection, (0,0), 3)  
        return np.clip(0.6*frame + 0.4*reflection, 0, 255)  

    def generate_frames(self, duration, fps):  
        """Temporal-coherent animation sequence"""  
        return [self._sakura_effect(self.base, i/fps).astype(np.uint8)  
                for i in range(int(duration*fps))]  

def main():  
    st.set_page_config(page_title="Sakura Animator", layout="wide")  

    with st.sidebar:  
        st.header("Cherry Blossom Controls")  
        duration = st.slider("Animation Duration (s)", 5.0, 30.0, 15.0)  
        fps = st.select_slider("Frame Rate", [24, 30, 60], 30)  

    uploaded = st.file_uploader("Upload Anime Background", type=["png", "jpg"])  

    if uploaded:  
        with st.spinner("Brewing matcha while generating..."):  
            img = Image.open(uploaded).convert("RGB")  
            engine = SakuraAnimationEngine(img)  

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:  
                frames = engine.generate_frames(duration, fps)  
                imageio.mimwrite(tmp.name, frames, fps=fps,  
                               codec="libx264", output_params=["-crf", "20"])  

                st.video(tmp.name)  
                st.download_button("Download Sakura Magic", tmp.name,  
                                 file_name="sakura_scene.mp4",  
                                 mime="video/mp4")  

if __name__ == "__main__":  
    main()  
