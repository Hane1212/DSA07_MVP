import streamlit as st
from PIL import Image

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
if uploaded_files and st.button("🔍 Run Detection"):
    st.subheader("Detection Results")
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=file.name, use_column_width=True)
        st.success("✅ Mock result: 12 fruits detected.")

# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)
