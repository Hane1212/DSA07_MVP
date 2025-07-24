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
from utils import enhanced_dashboard as enc

# Import enhanced modules with error handling
try:
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

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
if ENHANCED_MODULES_AVAILABLE:
    tab_detection, tab_chat, tab_his, tab_est, tab_enhanced = st.tabs([
        "🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs", "📈 Estimate", "🏆 Enhanced Analysis"
    ])
else:
    tab_detection, tab_chat, tab_his, tab_est = st.tabs([
        "🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs", "📈 Estimate"
    ])

# --- Initialize default values for export quality and price forecasting ---
# These should be calculated based on actual detection results
size = "Medium"  # Default value
ripeness = "Semi-Ripe"  # Default value
harvest_date = "2025-08-01"  # Default value
price = 45.50  # Default price in ₹ per kg
revenue = 2275.00  # Default revenue

# Check if we have actual detection results to update these values
if 'latest_detection_result' in st.session_state:
    result = st.session_state.latest_detection_result
    detections = result.get("detections", [])
    if detections:
        # Extract bounding boxes for size estimation
        bboxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3]) for d in detections]
        size = estimate.estimate_fruit_size(bboxes)
        
        # Get crop price and calculate revenue
        fruit_count = len(detections)
        avg_weight_grams = 150  # Default weight per fruit
        estimated_yield_kg = (fruit_count * avg_weight_grams) / 1000
        price = estimate.get_crop_price("Apple") or 45.50
        revenue = estimated_yield_kg * price

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

# --- TAB 4: Estimate ---
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

# --- TAB 5: Enhanced Analysis ---
enc.enhanced_analysis(ENHANCED_MODULES_AVAILABLE, tab_enhanced)

# --- Enhanced Results Display ---
enc.enhanced_results_display(ENHANCED_MODULES_AVAILABLE)

# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)