import streamlit as st
import requests
import os
from PIL import Image

# Title of the app
st.title("Image to Video AI App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    # Generate video on button click
    if st.button("Generate Video"):
        st.write("Generating video...")

        # Save the uploaded image temporarily
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Call the AI model API (e.g., Runway ML or custom backend)
        try:
            # Replace with your AI model API endpoint
            API_URL = "https://api.runwayml.com/v1/generate"
            API_KEY = "your_runway_api_key"

            headers = {"Authorization": f"Bearer {API_KEY}"}
            files = {"image": open(image_path, "rb")}
            response = requests.post(API_URL, headers=headers, files=files)

            if response.status_code == 200:
                # Save the generated video
                video_path = "generated_video.mp4"
                with open(video_path, "wb") as f:
                    f.write(response.content)

                # Display the generated video
                st.video(video_path)
                st.success("Video generated successfully!")

                # Clean up temporary files
                os.remove(image_path)
                os.remove(video_path)
            else:
                st.error("Failed to generate video. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
