import streamlit as st
import cv2
from models.yolo_detect import predict_fruit_count
import tempfile
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Real-time Fruit Counter", layout="wide")
st.title("📷 Real-time Fruit Counting")

# Streamlit camera input
img_data = st.camera_input("Take a picture")

if img_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_data.getbuffer())
        tmp_path = tmp.name

    image = cv2.imread(tmp_path)
    st.image(image, caption="📸 Captured Image", use_container_width=True)

    # Save to 'uploads' folder
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.basename(tmp_path)
    saved_path = os.path.join("uploads", filename)
    cv2.imwrite(saved_path, image)

    # Run YOLO detection
    st.subheader("🔍 YOLO Detection Result")
    results = predict_fruit_count(saved_path, return_results=True)
    boxed_img = results[0].plot()
    count = len(results[0].boxes)
    st.image(boxed_img, caption=f"✅ {count} fruits detected", use_container_width=True)
