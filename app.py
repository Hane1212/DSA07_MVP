import streamlit as st
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import requests
from utils.chatbox import show_chatbox
from utils.style import custom_css
from utils import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")

# --- PAGE SWITCH FUNCTION (REQUIRES streamlit>=1.12) ---
def go_to(page_name: str):
    st.switch_page(f"pages/{page_name}")

# --- CUSTOM CSS STYLE ---
st.markdown(custom_css, unsafe_allow_html=True)

# --- Tabs ---
tab_detection, tab_chat, tab_his = st.tabs(["🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs"])

# --- TAB 1: Detection ---
with tab_detection:
    st.header("🍌 Fruit Detection")

    model = st.selectbox("Select Model", ["YOLOv10m", "YOLOv9", "YOLOv8", "FasterCNN"])
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files and st.button("🔍 Run Detection"):
        st.subheader("Detection Results")
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            st.image(image, caption=f"Uploaded: {file.name}", width=int(image.width * 0.3))

            with st.spinner(f"Detecting using {model}..."):
                try:
                    result = utils.run_detection_api(file, model)
                    st.session_state[f"result_{file.name}"] = result  # ✅ store result by filename

                    st.success(f"✅ {result['fruit_count']} fruits detected.")
                    st.write(f"🔍 Model: {model}")
                    st.write(f"📏 Confidence: {result['confidence'] * 100:.2f}%")
                    st.write(f"⏱️ Detection time: {result['detection_time']} sec")
                    st.image(f"output/images/{file.name}", caption="Prediction Output", width=int(image.width * 0.3))

                except Exception as detect_err:
                    st.error("Prediction failed!")
                    st.exception(detect_err)

    # ✅ Now, after detection, show Save buttons
    if uploaded_files:
        st.subheader("Save Results to Database")
        for i, file in enumerate(uploaded_files):
            result_key = f"result_{file.name}"
            if result_key in st.session_state:
                result = st.session_state[result_key]
                if st.button(f"💾 Save", key=f"save-btn-{i}"):
                    try:
                        save_response = utils.save_detection_api(result, model)
                        st.success(f"✅ {file.name} saved to database.")
                    except Exception as save_err:
                        st.error(f"❌ Failed to save {file.name}.")
                        st.exception(save_err)

# --- TAB 2: Chatbot ---
with tab_chat:
    st.header("👩‍🌾 Ask AgriBot")
    show_chatbox()

# --- TAB 3: Historical Data ---
with tab_his:
    st.subheader("📜 Detection History")
    utils.show_detection_history()

# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)
