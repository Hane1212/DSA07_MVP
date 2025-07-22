import streamlit as st
import requests
from PIL import Image
import io
import os
# Removed: numpy, pandas, altair (as dashboard logic moved to _Compare.py)

# Import FastAPI app from utils.api.py (Note: We no longer run it from here)
# from utils.api import fastapi_app # No longer needed for direct import/run
from utils.style import custom_css
from utils.detection_utils import draw_boxes_on_image # Only need this for the home page

# IMPORTANT: Use the service name 'fastapi' as defined in docker-compose.yml
# for inter-container communication.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000") 

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Detection MVP", layout="wide")

# --- CUSTOM CSS STYLE ---
st.markdown(custom_css, unsafe_allow_html=True)

# --- Session State Initialization for app.py specific variables ---
# These are for the main detection page, distinct from compare page's state
if 'home_predictions' not in st.session_state:
    st.session_state.home_predictions = None
if 'home_original_image_bytes' not in st.session_state:
    st.session_state.home_original_image_bytes = None
if 'home_current_image_filename' not in st.session_state:
    st.session_state.home_current_image_filename = None

# --- Navigation ---
st.sidebar.title("Navigation")
# Use st.page_link for navigation to separate pages
st.sidebar.page_link("app.py", label="🏠 Image Prediction")
st.sidebar.page_link("pages/_Compare.py", label="📊 Compare Models")
st.sidebar.page_link("pages/_History.py", label="📜 History (Placeholder)") # Assuming _History.py exists or will be created

# --- Page Rendering Functions for app.py ---

def render_detection_tab():
    st.header("🍌 Fruit Detection")
    st.markdown("Upload an image and detect fruits using a single selected model.")

    model_choice = st.selectbox(
        "Select Model",
        ["YOLOv10m", "YOLOv9c", "YOLOv10l", "FasterRCNN"], # Added YOLOv10l
        index=0
    )

    uploaded_file = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=False)

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")

        if st.button("🔍 Run Detection"):
            with st.spinner(f"Detecting using {model_choice}..."):
                try:
                    data = {
                        "model_names": [model_choice], # Send as a list for single prediction
                        "image_filename": uploaded_file.name
                    }
                    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    
                    predict_url = f"{FASTAPI_URL}/predict" 
                    
                    response = requests.post(predict_url, data=data, files=files)
                    response.raise_for_status() 
                    
                    predictions = response.json()
                    st.session_state.home_predictions = predictions # Store for home page

                    st.subheader("Prediction Results:")

                    if model_choice in predictions:
                        detections = predictions[model_choice]
                        if detections:
                            st.write(f"Detected **{len(detections)}** objects:")
                            for i, det in enumerate(detections):
                                st.write(f"- Object {i+1}: Label: **{det['label']}**, Score: {det['score']:.2f}")
                            
                            img_with_boxes = image.copy()
                            img_with_boxes = draw_boxes_on_image(img_with_boxes, detections, model_choice)
                            st.image(img_with_boxes, caption=f"{model_choice} Detections", use_column_width=True)
                        else:
                            st.write(f"No objects detected by {model_choice} model.")
                    elif f"{model_choice}_error" in predictions:
                        st.error(f"{model_choice} Model Error: {predictions[f'{model_choice}_error']}")
                    else:
                        st.warning(f"No results or errors for {model_choice} model.")

                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure the server is running and accessible.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred during prediction: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please upload an image to get started.")

def render_chat_tab():
    st.header("👩‍🌾 Ask AgriBot")
    st.write("This is where the AgriBot chatbot functionality will be implemented.")
    st.info("Chatbot feature coming soon!")


# --- Main Application Logic (app.py) ---
# Use st.tabs for top-level navigation
tab_detection, tab_chat = st.tabs(["🧠 Detection", "💬 AgriBot Chat"]) # Removed compare tab from here

with tab_detection:
    render_detection_tab()

with tab_chat:
    render_chat_tab()

