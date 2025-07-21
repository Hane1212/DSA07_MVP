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
from utils.chatbox import show_chatbox
from utils.api import fastapi_app
from utils.style import custom_css

LOCAL_HOST = "http://localhost:8000/"

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")

# --- PAGE SWITCH FUNCTION (REQUIRES streamlit>=1.12) ---
def go_to(page_name: str):
    st.switch_page(f"pages/{page_name}")

def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

# --- CUSTOM CSS STYLE ---
st.markdown(custom_css, unsafe_allow_html=True)

# --- Tabs ---
tab_detection, tab_chat = st.tabs(["🧠 Detection", "💬 AgriBot Chat"])

# --- TAB 1: Detection ---
with tab_detection:
    st.header("🍌 Fruit Detection")

    model = st.selectbox("Select Model", ["YOLOv10m", "YOLOv9", "YOLOv8", "FasterCNN"])
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

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
                response = requests.post(f"{LOCAL_HOST}/predict/", files=files, data=data)
                result = response.json()
                st.success(f"✅ {result['fruit_count']} fruits detected.")
                st.write(f"🔍 Model: {model}")
                st.write(f"📏 Confidence: {result['confidence'] * 100:.2f}%")
                st.write(f"⏱️ Detection time: {result['detection_time']} sec")
                st.image("annotated.jpg", caption="Prediction Output", use_container_width=True)

            except Exception as e:
                st.error("Prediction failed!")
                st.exception(e)


# --- TAB 2: Chatbot ---
with tab_chat:
    st.header("👩‍🌾 Ask AgriBot")
    show_chatbox()
# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)
