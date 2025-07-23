import streamlit as st
import os
import requests


# for inter-container communication.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000") 

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "" # "admin", "annotator", "viewer"
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = ""

# ----------------------------------------
# Login page - Bhagyasri Parupudi &  Sree charan Lagudu
# ----------------------------------------
def render_authentication_page():
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


render_authentication_page()