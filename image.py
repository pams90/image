import streamlit as st
import requests
import os
from PIL import Image
import logging
import atexit

# Title of the app
st.title("Image to Video AI App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate video on button click
    if st.button("Generate Video"):
        st.write("Generating video...")

        # Save the uploaded image temporarily
        image_path = "temp_image.jpg"
        video_path = "generated_video.mp4"
        image.save(image_path)

        # Register cleanup function to remove temporary files
        def remove_temp_files():
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(video_path):
                os.remove(video_path)

        atexit.register(remove_temp_files)

        # Call the AI model API (e.g., Runway ML or custom backend)
        try:
            # Replace with your AI model API endpoint
            API_URL = "https://api.runwayml.com/v1/specific-endpoint"  # Update to the correct endpoint
            API_KEY = os.getenv("RUNWAY_API_KEY")

            if not API_KEY:
                st.error("API key is missing. Please set the RUNWAY_API_KEY environment variable.")
                logging.error("API key is missing.")
                st.stop()

            headers = {"Authorization": f"Bearer {API_KEY}"}
            files = {"image": open(image_path, "rb")}
            response = requests.post(API_URL, headers=headers, files=files)
            response.raise_for_status()

            # Save the generated video
            with open(video_path, "wb") as f:
                f.write(response.content)

            # Display the generated video
            st.video(video_path)
            st.success("Video generated successfully!")

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            st.error(f"An error occurred: {err}")
            logging.error(f"An error occurred: {err}")
