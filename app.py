import streamlit as st
from PIL import Image
from models.yolo_detect import predict_fruit_count
from utils.chatbox import show_chatbox
from utils.style import custom_css
from utils import utils
import os
import cv2
import numpy as np
from io import BytesIO
from streamlit_cropper import st_cropper

# --- Gamma Correction ---
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")
st.markdown(custom_css, unsafe_allow_html=True)
st.title("🍎 Fruit Counter - YOLOv10")

# --- Tabs ---
tab_detection, tab_chat, tab_his = st.tabs(["🧠 Detection", "💬 AgriBot Chat", "🗞️ History / Logs"])

# === TAB 1: Detection ===
with tab_detection:
    st.header("🍌 Fruit Detection")

    model = st.selectbox("Select Model", ["YOLOv10m", "YOLOv10l", "YOLOv9", "YOLOv8", "FasterCNN"])
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_key = f"image_bytes_{file.name}"
            if file_key not in st.session_state:
                st.session_state[file_key] = file.read()

            file_bytes = st.session_state[file_key]
            pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
            image_np = np.array(pil_image)

            # Brightness Adjustment
            brightness = st.slider(f"💡 Brightness for {file.name}", 0.5, 2.5, 1.0, 0.1,
                                   key=f"brightness_slider_{file.name}")
            bright_np = adjust_gamma(image_np, gamma=brightness)
            bright_pil = Image.fromarray(bright_np)

            if st.button(f"✅ Apply Brightness to {file.name}", key=f"apply_btn_{file.name}"):
                st.session_state[f"bright_{file.name}"] = bright_pil
                st.session_state[f"brightness_applied_{file.name}"] = True

            if st.session_state.get(f"brightness_applied_{file.name}", False):
                st.subheader("✂️ Crop Image (optional)")
                cropped_image = st_cropper(
                    st.session_state[f"bright_{file.name}"],
                    realtime_update=True,
                    box_color="#FF4B4B",
                    aspect_ratio=None,
                    key=f"cropper_{file.name}"
                )

                cropped_np = np.array(cropped_image.convert("RGB"))
                os.makedirs("uploads", exist_ok=True)
                image_path = os.path.join("uploads", file.name)
                cv2.imwrite(image_path, cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))

                st.session_state[f"path_{file.name}"] = image_path
                st.image(cropped_image, caption="🖼️ Final Cropped Image", use_container_width=True)

                # Run Detection
                if st.button(f"🔍 Run Detection on {file.name}", key=f"detect_btn_{file.name}"):
                    with st.spinner("Detecting..."):
                        results = predict_fruit_count(image_path, return_results=True)
                        count = len(results[0].boxes)
                        st.session_state[f"predicted_count_{file.name}"] = count

                        # Save boxed image
                        boxed_img = results[0].plot()
                        boxed_img_path = os.path.join("uploads", f"boxed_{file.name}")
                        cv2.imwrite(boxed_img_path, cv2.cvtColor(boxed_img, cv2.COLOR_RGB2BGR))

                        # Store in session
                        st.session_state[f"boxed_path_{file.name}"] = boxed_img_path
                        st.session_state[f"boxed_img_{file.name}"] = boxed_img

                        st.image(boxed_img, caption="🔍 Detection Result with Bounding Boxes", use_container_width=True)
                        st.success(f"✅ {count} fruits detected.")

                # Show image even after save or rerun
                boxed_img = st.session_state.get(f"boxed_img_{file.name}")
                if boxed_img is not None:
                    st.image(boxed_img, caption="🔍 Detection Result with Bounding Boxes", use_container_width=True)

                pred = st.session_state.get(f"predicted_count_{file.name}")
                if pred is not None:
                    st.success(f"✅ {pred} fruits detected.")
                    corrected = st.number_input(
                        f"✏️ Correct count for {file.name}:", 0, pred + 10, pred, 1,
                        key=f"corrected_{file.name}"
                    )

                    if st.button(f"📂 Save Result for {file.name}", key=f"save_btn_{file.name}"):
                        result = {
                            "image_id": file.name,
                            "image_path": st.session_state.get(f"path_{file.name}"),
                            "boxed_path": st.session_state.get(f"boxed_path_{file.name}"),
                            "fruit_count": pred,
                            "corrected_count": corrected,
                            "model_used": model
                        }
                        utils.save_to_csv(result, model)
                        st.success("✅ Result saved to history.")
                        st.rerun()

    # --- Download Report Button Moved Here ---
    st.subheader("📥 Export Full Report")
    if st.button("📄 Download PDF Report", key="download_pdf_button_top"):
        try:
            report_path = utils.generate_pdf_from_csv()
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Full Report",
                    data=f,
                    file_name="fruit_report.pdf",
                    mime="application/pdf",
                    key="download_pdf_file_top"
                )
        except Exception as e:
            st.error("❌ Failed to generate PDF.")
            st.exception(e)

# === TAB 2: Chatbot ===
with tab_chat:
    st.header("👩‍🌾 Ask AgriBot")
    show_chatbox()

# === TAB 3: History / Logs ===
with tab_his:
    st.subheader("📜 Detection History")
    utils.show_detection_history()

# --- CLOSE ---
st.markdown("</div>", unsafe_allow_html=True)
