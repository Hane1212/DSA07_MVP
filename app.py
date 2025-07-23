import streamlit as st
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from utils.chatbox import show_chatbox
from utils.style import custom_css
from utils import estimate
from utils import utils

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")

# --- PAGE SWITCH FUNCTION (REQUIRES streamlit>=1.12) ---
def go_to(page_name: str):
    st.switch_page(f"pages/{page_name}")

# --- CUSTOM CSS STYLE ---
st.markdown(custom_css, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "" # "admin", "annotator", "viewer"
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = ""

# --- Navigation ---
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/_Compare.py", label="📊 Compare Models")
st.sidebar.page_link("pages/_Login.py", label="🔑 User Authentication")
# --- Tabs ---
tab_detection, tab_chat, tab_his, tab_est = st.tabs(["🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs", "📈 Estimate"])

# --- TAB 1: Detection ---
with tab_detection:
    st.header("Fruit Detection")
    utils.show_fruit_detection()

# --- TAB 2: Chatbot ---
with tab_chat:
    st.header("👩‍🌾 Ask AgriBot")
    show_chatbox()

# --- TAB 3: Historical Data ---
with tab_his:
    st.subheader("📜 Detection History")
    utils.show_detection_history()

with tab_est:
    st.subheader("🧮 Estimate")
    for key in st.session_state:
        if key.startswith("result_"):
            file_name = key.replace("result_", "")
            result = st.session_state[key]
            uploaded_file = st.session_state.get(f"file_{file_name}")
            image = st.session_state.get(f"image_{file_name}")
            
            if uploaded_file and image:
                estimate.show_estimate_for_file(uploaded_file, result)


# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)
