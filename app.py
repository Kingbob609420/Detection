# 🚨 IMPORTS FIRST
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# 🚨 SET PAGE CONFIG IMMEDIATELY AFTER IMPORTS
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")

# 🚨 THEN EVERYTHING ELSE (Functions, Streamlit commands)

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Make sure yolov5 repo is local
    return model

model = load_model()

# Helper to detect objects in an uploaded image
def detect_objects_image(uploaded_image):
    img = Image.open(uploaded_image)
    img_array = np.array(img)
    results = model(img_array)
    results.render()
    detected_img = Image.fromarray(results.ims[0])
    return detected_img

# Helper to detect objects from webcam frame
def detect_objects_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    results.render()
    img_bgr = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    return img_bgr

# 🚨 STREAMLIT UI STARTS
st.title("🛡️ YOLOv5 - Object Detection Web App")

# Tabs for Image and Webcam
tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Webcam Detection"])

# ========== IMAGE DETECTION ==========
with tab1:
    st.header("Upload an Image for Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting..."):
                result_img = detect_objects_image(uploaded_file)
                st.image(result_img, caption="Detected Image", use_column_width=True)
            st.success("Detection Complete!")

# ========== WEBCAM DETECTION ==========
with tab2:
    st.header("Live Webcam Object Detection")
    run_webcam = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run_webcam:
        cap = cv2.VideoCapture(0)
        st.info("Webcam started. Press 'Stop Webcam' to stop.")

        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam.")
                break

            frame = detect_objects_frame(frame)
            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

    else:
        st.write("Webcam is stopped.")
