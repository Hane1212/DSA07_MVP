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

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "" # "admin", "annotator", "viewer"
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = ""

# --- Helper to get auth headers ---
def get_auth_headers():
    if st.session_state.logged_in and st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}

# --- Page Rendering Functions ---

def render_authentication_tab():
    st.header("🔑 User Authentication")

    auth_option = st.radio("Choose an option", ["Login", "Register"], key="auth_option")

    if auth_option == "Login":
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if not login_username or not login_password:
                st.error("Please enter both username and password.")
                return

            try:
                response = requests.post(
                    f"{FASTAPI_URL}/auth/login",
                    json={"username": login_username, "password": login_password}
                )
                response.raise_for_status()
                token_data = response.json()
                
                st.session_state.logged_in = True
                st.session_state.username = token_data["username"]
                st.session_state.user_role = token_data["role"]
                st.session_state.auth_token = token_data["access_token"]
                st.success(f"Logged in as {st.session_state.username} ({st.session_state.user_role})")
                st.rerun() # Rerun to update UI based on login state

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Invalid username or password.")
                else:
                    st.error(f"Login failed: {e.response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure the server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during login: {e}")

    elif auth_option == "Register":
        st.subheader("Register New User")
        reg_username = st.text_input("New Username", key="reg_username")
        reg_password = st.text_input("New Password", type="password", key="reg_password")
        reg_role = st.selectbox("Select Role", ["viewer", "annotator", "admin"], key="reg_role") # Allow registration of different roles for demo

        if st.button("Register"):
            if not reg_username or not reg_password:
                st.error("Please enter username and password.")
                return

            try:
                response = requests.post(
                    f"{FASTAPI_URL}/auth/register",
                    json={"username": reg_username, "password": reg_password, "role": reg_role}
                )
                response.raise_for_status()
                st.success(f"User '{reg_username}' registered successfully as '{reg_role}'. You can now log in.")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    st.error(f"Registration failed: {e.response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure the server is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred during registration: {e}")

    if st.session_state.logged_in:
        st.markdown("---")
        st.success(f"Currently logged in as **{st.session_state.username}** with role **{st.session_state.user_role}**.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_role = ""
            st.session_state.auth_token = ""
            st.info("Logged out successfully.")
            st.rerun()

def render_detection_tab():
    st.header("🍌 Fruit Detection")
    st.markdown("Upload an image and detect fruits using a single selected model.")

    if not st.session_state.logged_in:
        st.warning("Please log in to use the detection feature.")
        return

    model_choice = st.selectbox(
        "Select Model",
        ["YOLOv10m", "YOLOv9c", "YOLOv10l", "FasterRCNN"], 
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
                        "model_names": [model_choice], 
                        "image_filename": uploaded_file.name
                    }
                    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    
                    predict_url = f"{FASTAPI_URL}/predict" 
                    
                    response = requests.post(predict_url, data=data, files=files, headers=get_auth_headers())
                    response.raise_for_status() 
                    
                    predictions = response.json()
                    st.session_state.home_predictions = predictions 

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

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        st.error("Authentication required. Please log in.")
                    elif e.response.status_code == 403:
                        st.error("You do not have permission to perform this action.")
                    else:
                        st.error(f"Prediction failed: {e.response.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure the server is running.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Please upload an image to get started.")

    # Admin/Annotator specific actions
    st.markdown("---")
    st.subheader("Admin/Annotator Actions (Placeholder)")
    if st.session_state.user_role == "admin":
        if st.button("📊 Train Models (Admin Only)"):
            try:
                response = requests.post(f"{FASTAPI_URL}/admin/train", headers=get_auth_headers())
                response.raise_for_status()
                st.success(response.json().get("message", "Training initiated."))
            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to train models: {e.response.json().get('detail', 'Unknown error')}")
        if st.button("🗑️ Delete Records (Admin Only)"):
            record_id_to_delete = st.text_input("Record ID to Delete", key="delete_record_id")
            if st.button("Confirm Delete"):
                try:
                    response = requests.delete(f"{FASTAPI_URL}/admin/records/{record_id_to_delete}", headers=get_auth_headers())
                    response.raise_for_status()
                    st.success(response.json().get("message", "Record deleted."))
                except requests.exceptions.HTTPError as e:
                    st.error(f"Failed to delete record: {e.response.json().get('detail', 'Unknown error')}")
    elif st.session_state.user_role == "annotator":
        if st.button("✍️ Correct Predictions (Annotator Only)"):
            try:
                response = requests.post(f"{FASTAPI_URL}/annotator/correct_prediction", headers=get_auth_headers())
                response.raise_for_status()
                st.success(response.json().get("message", "Correction initiated."))
            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to correct predictions: {e.response.json().get('detail', 'Unknown error')}")
    else:
        st.info("Additional actions are available for 'annotator' and 'admin' roles.")


def render_chat_tab():
    st.header("👩‍🌾 Ask AgriBot")
    st.write("This is where the AgriBot chatbot functionality will be implemented.")
    st.info("Chatbot feature coming soon!")


# --- Main Application Logic (app.py) ---
# Use st.tabs for top-level navigation
tab_auth, tab_detection, tab_chat = st.tabs(["🔑 Authentication", "🧠 Detection", "💬 AgriBot Chat"]) 

with tab_auth:
    render_authentication_tab()

with tab_detection:
    render_detection_tab()

with tab_chat:
    render_chat_tab()

