import streamlit as st
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import time
import threading
from ultralytics import YOLO
import cv2
import numpy as np
import os

# --- FastAPI Backend ---
fastapi_app = FastAPI()

# Allow Streamlit frontend to call this backend
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv10 model once
yolo_model = YOLO("yolov10m.pt") 

@fastapi_app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    start = time.time()
    
    # Read image
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run model
    results = yolo_model(image)[0]

    # Count detections
    boxes = results.boxes
    fruit_count = len(boxes)

    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save annotated image
    output_path = "annotated.jpg"
    cv2.imwrite(output_path, image)

    detection_time = round(time.time() - start, 2)
    return JSONResponse(content={
        "fruit_count": fruit_count,
        "confidence": float(boxes[0].conf[0]) if fruit_count > 0 else 0,
        "detection_time": detection_time
    })
# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")

# --- CUSTOM CSS STYLE ---
st.markdown("""
    <style>
    body {
        background-color: #c7dab5;
    }
    .main-container {
        background-color: #fceee3;
        padding: 3rem;
        border-radius: 20px;
        width: 50%;
        margin: auto;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
    }
    .button-row {
        position: absolute;
        top: 1rem;
        right: 2rem;
    }
    .button-row a {
        padding: 0.5rem 1rem;
        background-color: #648b5b;
        color: white;
        border-radius: 6px;
        margin-left: 0.5rem;
        text-decoration: none;
        font-weight: bold;
    }
    .header {
        text-align: center;
    }
    .header h1 {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .header p {
        font-size: 1.2rem;
        color: #555;
    }
    </style>

    <div class="button-row">
        <a href="#">Compare</a>
        <a href="#">History</a>
        <a href="#">🔐 Login</a>
    </div>

    <div class="main-container">
        <div class="header">
            <h1>🍎 Fruit Counter MVP</h1>
            <p>Detect and count fruits in orchard images using AI</p>
        </div>
""", unsafe_allow_html=True)

# --- SELECT MODEL ---
model = st.selectbox("**Select Model**", ["YOLOv10m", "YOLOv9", "YOLOv8", "FasterCNN"])

# --- IMAGE UPLOAD ---
uploaded_files = st.file_uploader("Choose Images", type=["jpg", "png"], accept_multiple_files=True)

# --- DETECTION (Mock) ---
import requests

if uploaded_files and st.button("🔍 Run Detection"):
    st.subheader("Detection Results")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=f"Uploaded: {file.name}", use_container_width=True)

        # Send image + model to backend
        with st.spinner(f"Detecting using {model}..."):
            files = {"file": (file.name, file.getvalue(), "image/png")}
            data = {"model_name": model}
            try:
                response = requests.post("http://localhost:8000/predict/", files=files, data=data)
                result = response.json()
                st.success(f"✅ {result['fruit_count']} fruits detected.")
                st.write(f"🔍 Model: {model}")
                st.write(f"📏 Confidence: {result['confidence'] * 100:.2f}%")
                st.write(f"⏱️ Detection time: {result['detection_time']} sec")
                st.image("annotated.jpg", caption="Prediction Output", use_container_width=True)

            except Exception as e:
                st.error("Prediction failed!")
                st.exception(e)


# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)
# --- Run FastAPI and Streamlit in parallel ---
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# Start FastAPI in a background thread
threading.Thread(target=run_fastapi, daemon=True).start()
