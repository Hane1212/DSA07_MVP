import requests
import streamlit as st
import pandas as pd
import torchvision
import torch
import io
import cv2
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from streamlit_cropper import st_cropper
import numpy as np

LOCAL_HOST = "http://fastapi:8000/"

"""
Author: Huong TA
Date: 2025-07-22
Description: Call predict API 
"""
def run_detection_api(uploaded_files, model_name):
    if uploaded_files and st.button("🔍 Run Detection"):
        st.subheader("Detection Results")
        for i, file in enumerate(uploaded_files):
            files = {"file": (file.name, file.getvalue(), "image/png")}
            data = {"model_name": model_name}
            image = Image.open(file)
            with st.spinner(f"Detecting using {model_name}..."):
                try:
                    response = requests.post(f"{LOCAL_HOST}/predict/", files=files, data=data)
                    response.raise_for_status()
                    result = response.json()
                    result['model_name'] = model_name
                    st.session_state[f"result_{file.name}"] = result
                    st.session_state[f"file_{file.name}"] = file
                    st.session_state[f"image_{file.name}"] = image
                    st.session_state[f"predicted_image_{file.name}"] = f"output/images/{file.name}"
                    st.success(f"✅ {result['fruit_count']} fruits detected.")
                except Exception as detect_err:
                    st.error("Prediction failed!")
                    st.exception(detect_err)


def display_detection_results(uploaded_files):
    for i, file in enumerate(uploaded_files):
        result_key = f"result_{file.name}"
        if result_key in st.session_state:
            result = st.session_state[result_key]
            image_path = st.session_state.get(f"predicted_image_{file.name}", None)
            if image_path:
                st.image(image_path, caption=f"🖼️ Prediction: {file.name}", width=300)

            st.write(f"🔍 Model: {result['model_name']}")
            st.write(f"📏 Confidence: {result['confidence']*100:.2f}%")
            st.write(f"⏱️ Detection time: {result['detection_time']} sec")
            edit_count_number(file, result, unique_prefix=f"row_{i}")


def edit_count_number(file, result, unique_prefix=""):
    safe_key = f"{unique_prefix}_{file.name}"  # Make key globally unique
    edit_flag_key = f"edit_mode_enabled_{safe_key}"
    input_key = f"input_{safe_key}"
    confirm_key = f"confirm_edit_{safe_key}"
    corrected_key = f"corrected_count_{safe_key}"


    if st.button(f"✏️ Edit number of fruit", key=f"edit_btn_{safe_key}"):
        st.session_state[edit_flag_key] = True

    if st.session_state.get(edit_flag_key, False):
        new_count = st.number_input(
            f"📝 Corrected number of fruits for {file.name}",
            min_value=0,
            value=result['fruit_count'],
            key=input_key
        )

        if st.button(f"✅ Confirm", key=confirm_key):
            st.session_state[corrected_key] = new_count
            result["num_objects_corrected"] = new_count
            result["annotated_by_user"] = True
            st.success(f"Confirmed: corrected count = {new_count}")


"""
Author: Huong TA
Date: 2025-07-22
Description: Call save API to save data to DB 
"""
def save_detection_api(result, model_name, file):
    """
    Sends detection metadata to the /save endpoint.
    """
    print("📤 Sending save request...")
    payload = {
        "image_id": result["image_id"],
        "image_path": result["image_path"],
        "model_name": model_name,
        "detection_time": result["detection_time"],
        "confidence": result["confidence"],
        "model_predictions": result.get("model_predictions", []),
        "user_annotations": [],
        "num_objects_model": result["fruit_count"],
        "num_objects_corrected": result.get("num_objects_corrected", 0),
        "annotated_by_user": result.get("annotated_by_user", False),
        "annotator_id": None
    }

    response = requests.post(f"{LOCAL_HOST}/save", json=payload)
    print("📥 Response:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()


"""
Author: Huong TA
Date: 2025-07-22
Description: call detections API to get data from DB 
"""
def show_detection_history():
    with st.form("filter_form"):
        image_filter = st.text_input("🔍 Filter by image name (partial match)")
        min_count = st.number_input("🔢 Minimum object count", min_value=0, step=1)
        show_all = st.checkbox("📋 Show all (ignore filters)", value=False)
        submitted = st.form_submit_button("🖼️ Get Image")

    if submitted:
        try:
            params = {}
            if not show_all:
                if image_filter:
                    params["image_id"] = image_filter
                if min_count:
                    params["min_count"] = min_count

            response = requests.get(f"{LOCAL_HOST}/detections", params=params)
            data = response.json()

            if not data:
                st.info("No matching records found.")
                return

            # Build table
            rows = []
            for d in data:
                rows.append([
                    d.get("image_id"),
                    d.get("model_name"),
                    d.get("num_objects_model"),
                    d.get("num_objects_corrected"),
                    d.get("detection_time"),
                    f"{d.get('confidence') * 100:.2f}%"
                ])

            st.markdown("### 🧾 Results")
            st.table(
                pd.DataFrame(rows, columns=[
                    "🖼 Image Name", "🤖 Model", "🔢 Model Count", "✏️ Corrected Count",
                    "⏱ Detection Time (s)", "📏 Confidence"
                ])
            )

        except Exception as e:
            st.error("⚠️ Failed to load detection data.")
            st.exception(e)


"""
Author: Huong TA
Date: 2025-07-22
Description: Load faster_rcnn_resnet50 model
"""
NUM_CLASSES = 2  # 1 class: apple + 1 background
def load_faster_rcnn_resnet50(num_classes=NUM_CLASSES, path="./model/fasterrcnn_resNet50_fpn.pth"):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head with own number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load custom weights
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

"""
Author: Huong TA
Date: 2025-07-22
Description: load_faster_rcnn_mobilenet
"""
def load_faster_rcnn_mobilenet(num_classes=NUM_CLASSES, path="./model/fasterrcnn_mobilenet_fpn.pth"):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

"""
Author: Huong TA
Date: 2025-07-21
Description: Save result to DB
"""
def save_to_DB(uploaded_files, model_name):
    if uploaded_files:
        st.subheader("Save Results to Database")
        for i, file in enumerate(uploaded_files):
            result_key = f"result_{file.name}"
            if result_key in st.session_state:
                result = st.session_state[result_key]
                if st.button(f"💾 Save", key=f"save-btn-{i}"):
                    try:
                        save_detection_api(result, model_name, file)
                        st.success(f"✅ {file.name} saved to database.")
                    except Exception as save_err:
                        st.error(f"❌ Failed to save {file.name}.")
                        st.exception(save_err)


"""
Author: Vinay Gaddam
Date: 2025-07-23
Description: 
"""
# --- Gamma Correction ---
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def handle_edit_image(uploaded_files):
    for file in uploaded_files:
        file_key = f"image_bytes_{file.name}"
        file_bytes = st.session_state[file_key]  # Already stored

        pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_np = np.array(pil_image)

        st.markdown(f"### ✏️ Editing: {file.name}")
        col1, col2 = st.columns(2)

        # --- Column 1: Brightness ---
        with col1:
            st.markdown("#### 🔆 Brightness Adjustment")
            brightness = st.slider(f"Brightness", 0.5, 2.5, 1.0, 0.1, key=f"brightness_slider_{file.name}")
            bright_np = adjust_gamma(image_np, gamma=brightness)
            bright_pil = Image.fromarray(bright_np)
            display_width = int(pil_image.width * 0.3)
            st.image(bright_pil, caption="Preview: Brightness Adjusted", width=display_width)

        # --- Column 2: Cropping ---
        with col2:
            st.markdown("#### ✂️ Crop Image (interactive)")
            crop_col1, crop_col2 = st.columns([2, 1])  # Wider for cropper, narrower for preview

            with crop_col1:
                # Use bright_pil directly (ensure it’s a valid PIL Image)
                cropped_image = st_cropper(
                    bright_pil,
                    realtime_update=True,
                    box_color="#FF4B4B",
                    aspect_ratio=None,
                    key=f"cropper_{file.name}"
                )

            with crop_col2:
                if cropped_image:
                    st.image(cropped_image, caption="Preview: Cropped", width=int(bright_pil.width * 0.3))

        # --- Apply Combined Changes ---
        if st.button(f"✅ Apply Changes", key=f"apply_all_{file.name}"):
            cropped_np = np.array(cropped_image.convert("RGB"))
            os.makedirs("uploads", exist_ok=True)
            image_path = os.path.join("uploads", file.name)
            cv2.imwrite(image_path, cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))

            st.session_state[f"final_{file.name}"] = cropped_image
            st.session_state[f"path_{file.name}"] = image_path

            st.success(f"✅ Edits applied and saved for {file.name}")



"""
Author: Huong TA
Date: 2025-07-20
Description: Show Main screen
"""
def show_fruit_detection():
    col1, _, _ = st.columns([1, 1, 1])
    with col1:
        model = st.selectbox("Select Model", ["YOLOv10m", "YOLOv9", "fasterRCNNmobile", "fasterRCNNresNet50"])

    col2, _, _ = st.columns([1, 1, 1])
    with col2:
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)


    if uploaded_files:
        # Save image bytes once in session_state
        for file in uploaded_files:
            file_key = f"image_bytes_{file.name}"
            if file_key not in st.session_state:
                st.session_state[file_key] = file.read()
            # Edit button logic
            edit_key = f"edit_enabled_{file.name}"
            if st.button(f"✏️ Edit", key=f"edit_btn_{file.name}"):
                st.session_state[edit_key] = True

        # Handle editing if requested
        files_to_edit = [f for f in uploaded_files if st.session_state.get(f"edit_enabled_{f.name}", False)]
        if files_to_edit:
            handle_edit_image(files_to_edit)
            
    final_files = []
    for file in uploaded_files:
        edit_key = f"edit_enabled_{file.name}"
        if st.session_state.get(edit_key):
            final_img = st.session_state.get(f"final_{file.name}")
            if final_img:
                # Convert edited PIL image back to BytesIO with same name
                img_buffer = BytesIO()
                final_img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                img_buffer.name = file.name  # mimic UploadFile
                final_files.append(img_buffer)
            else:
                final_files.append(file)  # fallback
        else:
            final_files.append(file)
    
    # Call detection and save
    run_detection_api(final_files, model)
    display_detection_results(final_files)
    save_to_DB(final_files, model_name=model)

# ----------------------------------------
# API for compare page - Bhagyasri Parupudi
# ----------------------------------------
from fastapi import HTTPException, Form
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional as F
from ultralytics import YOLO
import csv
import pickle
import tempfile
import os
# --- Global variables for models and ground truth ---
yolo_models = {} # Stores {"YOLOv10m": model_obj, "YOLOv9c": model_obj, "YOLOv10l": model_obj}
cnn_resnet50_model = None
ground_truth_counts = {} # To store ground truth data

# Define class labels for your models.
# IMPORTANT: YOLO_CLASS_NAMES will now be set dynamically from the loaded YOLO model.
YOLO_CLASS_NAMES = {
    "YOLOv10m": [],
    "YOLOv9c": [],
    "YOLOv10l": [] # Added YOLOv10l
} 

# For Faster R-CNN, the convention is often that index 0 is '__background__'.
CNN_CLASS_NAMES = [
    "__background__", "fruit" # This matches 2 classes: 0 for background, 1 for 'fruit'
] 

# --- Ground Truth Loading Function ---
def load_ground_truth():
    """Loads ground truth counts from ground_truth.txt."""
    global ground_truth_counts
    print("Loading ground_truth.txt...")
    try:
        # Assuming ground_truth.txt is in the root or a known path
        with open('ground_truth.txt', 'r') as f: # Adjust path if necessary
            reader = csv.reader(f)
            header = next(reader) # Skip header row
            if header != ['Image', 'count']:
                print(f"WARNING: ground_truth.txt header mismatch. Expected ['Image', 'count'], got {header}")
            for row in reader:
                if len(row) == 2:
                    image_name, count = row
                    ground_truth_counts[image_name.strip()] = int(count.strip())
        print(f"Loaded {len(ground_truth_counts)} ground truth entries.")
    except FileNotFoundError:
        print("CRITICAL ERROR: ground_truth.txt not found. Ground truth metrics will not be available.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load ground_truth.txt: {e}")


# --- Helper function to draw bounding boxes ---
def draw_boxes_on_image(image: Image.Image, detections: list, model_name: str) -> Image.Image:
    """Draws bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default() # Fallback font

    # Assign distinct colors for different models
    color = "red"
    if model_name == "YOLOv10m":
        color = "red"
    elif model_name == "YOLOv9c":
        color = "green" # Different color for YOLOv9c
    elif model_name == "YOLOv10l": # Added YOLOv10l color
        color = "purple"
    elif model_name == "FasterRCNN":
        color = "blue"

    for det in detections:
        box = det['box']
        label = det['label']
        score = det['score']
        xmin, ymin, xmax, ymax = [int(b) for b in box]

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        text = f"{label} ({score:.2f})"
        text_width = draw.textlength(text, font) 
        text_height = font.getbbox(text)[3] - font.getbbox(text)[1] 
        
        # Draw text background
        draw.rectangle([xmin, ymin - text_height - 5, xmin + text_width + 5, ymin], fill=color)
        draw.text((xmin + 2, ymin - text_height - 3), text, fill="white", font=font)
    
    return image

# --- PDF Generation Function ---
def create_prediction_pdf(original_image_bytes, image_filename, predictions_data, ground_truth_count, metrics_df, combined_img_with_boxes=None):
    """
    Generates a PDF report for predictions.
    predictions_data is a dictionary like {"YOLOv10m": [...], "FasterRCNN": [...]}
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.write(5, "Fruit Detection Report\n")
    pdf.write(5, f"Image File: {image_filename}\n\n")

    # Add original image
    if original_image_bytes:
        pdf.write(5, "Original Image:\n")
        try:
            original_pil_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
            img_buffer_for_pdf = io.BytesIO()
            original_pil_image.save(img_buffer_for_pdf, format="PNG")
            img_buffer_for_pdf.seek(0)
            img_width, img_height = original_pil_image.size
            max_width = pdf.w - 2 * pdf.l_margin
            
            # Calculate image dimensions for PDF to fit page width while maintaining aspect ratio
            if img_width > max_width:
                pdf_image_width = max_width
                pdf_image_height = img_height * (max_width / img_width)
            else:
                # If image is smaller than max_width, scale it down to a reasonable size (e.g., 1/2 of page width)
                pdf_image_width = img_width * 0.5 # Adjust scaling factor as needed
                pdf_image_height = img_height * 0.5
                # Ensure it doesn't exceed max_width if scaled up too much
                if pdf_image_width > max_width:
                    pdf_image_width = max_width
                    pdf_image_height = img_height * (max_width / img_width)

            pdf.image(img_buffer_for_pdf, x=pdf.l_margin, y=pdf.get_y(), w=pdf_image_width, h=pdf_image_height, type="PNG")
            pdf.ln(pdf_image_height + 5)
        except Exception as e:
            pdf.set_text_color(255, 0, 0)
            pdf.write(5, f"Could not embed original image: {e}\n")
            pdf.set_text_color(0, 0, 0)

    # Add Combined Detections Image if available
    if combined_img_with_boxes:
        pdf.write(5, "\nCombined Detections:\n")
        img_buffer_combined = io.BytesIO()
        combined_img_with_boxes.save(img_buffer_combined, format="PNG")
        img_buffer_combined.seek(0)
        
        img_width, img_height = combined_img_with_boxes.size
        max_width = pdf.w - 2 * pdf.l_margin
        if img_width > max_width:
            pdf_image_width = max_width
            pdf_image_height = img_height * (max_width / img_width)
        else:
            pdf_image_width = img_width * 0.5 # Adjust scaling factor
            pdf_image_height = img_height * 0.5
            if pdf_image_width > max_width:
                pdf_image_width = max_width
                pdf_image_height = img_height * (max_width / img_width)

        # Save to temp file first - Huong TA
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_buffer_combined.getvalue())
            temp_filename = tmp_file.name
        # pdf.image(img_buffer_combined, x=pdf.l_margin, y=pdf.get_y(), w=pdf_image_width, h=pdf_image_height, type="PNG")
        pdf.image(temp_filename, x=pdf.l_margin, y=pdf.get_y(), w=pdf_image_width, h=pdf_image_height, type="PNG") # Huong TA
        pdf.ln(pdf_image_height + 5)
        os.remove(temp_filename)

    pdf.write(5, "\n--- Detections Summary ---\n")

    for model_name, detections in predictions_data.items():
        if isinstance(detections, list): # Ensure it's detection list, not error string
            pdf.set_font("Arial", 'B', 12)
            pdf.write(5, f"\n{model_name} Model Detected: {len(detections)} objects\n")
            pdf.set_font("Arial", size=10)
            for i, det in enumerate(detections):
                pdf.write(5, f"- Object {i+1}: Label: {det['label']}, Score: {det['score']:.2f}\n")
        elif isinstance(detections, str) and "error" in model_name.lower():
             pdf.set_text_color(255, 0, 0)
             pdf.write(5, f"\n{model_name.replace('_error', '')} Model Error: {detections}\n")
             pdf.set_text_color(0, 0, 0)

    pdf.ln(5)

    pdf.write(5, "\n--- Performance Metrics ---\n")
    pdf.set_font("Arial", size=10)

    if ground_truth_count is not None:
        pdf.write(5, f"Ground Truth Count for this image: {ground_truth_count}\n")
        for model_name, detections in predictions_data.items():
            if isinstance(detections, list):
                model_count = len(detections)
                count_accuracy = 1.0 - (abs(model_count - ground_truth_count) / max(ground_truth_count, 1))
                pdf.write(5, f"{model_name} Count Accuracy: {count_accuracy:.2f}\n")
    else:
        pdf.write(5, "Ground Truth Count: Not available for this image.\n")

    pdf.ln(5)

    # Add metrics table
    if metrics_df is not None and not metrics_df.empty:
        col_widths = [40] + [40] * (len(metrics_df.columns) - 1) # Dynamic column widths
        
        pdf.set_font("Arial", 'B', 10)
        for i, header in enumerate(metrics_df.columns):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()

        pdf.set_font("Arial", size=10)
        for index, row in metrics_df.iterrows():
            pdf.cell(col_widths[0], 10, str(row["Metric"]), 1)
            for i in range(1, len(metrics_df.columns)):
                col_name = metrics_df.columns[i]
                val = row[col_name]
                if row['Metric'] == "Number of Detections":
                    pdf.cell(col_widths[i], 10, str(int(val)), 1, 0, 'C')
                else:
                    pdf.cell(col_widths[i], 10, f"{val:.2f}", 1, 0, 'C')
            pdf.ln()
    else:
        pdf.write(5, "No comparison metrics available to display in table.\n")

    pdf.ln(10)
    pdf.set_font("Arial", size=8)
    pdf.write(5, "*Simulated metrics are for illustration only and require ground truth bounding boxes for actual calculation.")

    # return bytes(pdf.output(dest='S'))
    return pdf.output(dest='S').encode('latin1') # Huong TA

# --- Model Loading Functions ---
def load_yolo_model(model_name: str, model_path: str):
    """Loads a specific YOLO model from a .pt file."""
    global yolo_models
    global YOLO_CLASS_NAMES
    if model_name not in yolo_models or yolo_models[model_name] is None:
        print(f"Attempting to load YOLO model: {model_name} from {model_path}...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"YOLO model '{model_name}' will attempt to load on device: {device}")
            
            yolo_instance = YOLO(model_path) 
            yolo_instance.eval() # Set model to evaluation mode
            
            if hasattr(yolo_instance, 'names') and yolo_instance.names:
                YOLO_CLASS_NAMES[model_name] = yolo_instance.names # Use the model's actual class names
                print(f"YOLO model '{model_name}' internal class names: {YOLO_CLASS_NAMES[model_name]}")
            else:
                print(f"WARNING: YOLO model '{model_name}' does not have 'names' attribute. Using default placeholder names.")
            
            yolo_models[model_name] = yolo_instance
            print(f"YOLO model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load YOLO model '{model_name}': {e}")
            print(f"Detailed YOLO loading error for {model_name}: {e.__class__.__name__}: {e}")
            yolo_models[model_name] = None
    return yolo_models.get(model_name)

def load_cnn_resnet50_model():
    """Loads the Faster R-CNN ResNet50 model from a .pth file."""
    global cnn_resnet50_model
    if cnn_resnet50_model is None:
        print(f"Attempting to load Faster R-CNN (ResNet50) model...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"CNN model (ResNet50) will attempt to load on device: {device}")

            cnn_expected_num_classes = 2 # Background + Fruit
            
            model_path = 'models/fasterrcnn_resNet50_fpn.pth' # Adjust path if necessary
            cnn_instance = fasterrcnn_resnet50_fpn(weights=None, num_classes=cnn_expected_num_classes)

            print(f"Initializing Faster R-CNN (ResNet50) with num_classes={cnn_expected_num_classes}")
            
            print(f"Loading state_dict from '{model_path}' with map_location=torch.device('cpu')")
            state_dict = torch.load(
                model_path, 
                map_location=torch.device('cpu')
            )
            
            cnn_instance.load_state_dict(state_dict)
            cnn_instance.to(device) # Move model to the selected device
            cnn_instance.eval() # Set model to evaluation mode
            cnn_resnet50_model = cnn_instance
            print(f"Faster R-CNN (ResNet50) model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Faster R-CNN (ResNet50) model: {e}")
            print(f"Detailed CNN loading error: {e.__class__.__name__}: {e}")
            cnn_resnet50_model = None # Ensure it's None if loading fails
    return cnn_resnet50_model

# --- Model Prediction Functions ---

def predict_yolo(yolo_model_instance: YOLO, model_name_key: str, image: Image.Image):
    """
    Performs inference using a specific YOLO model instance.
    Returns a list of dictionaries, each representing a detected object.
    Each dict contains 'box' (xmin, ymin, xmax, ymax), 'score', and 'label'.
    """
    if yolo_model_instance is None:
        raise HTTPException(status_code=500, detail=f"YOLO model '{model_name_key}' not loaded. Check server logs for details.")

    try:
        print(f"Performing YOLO prediction with {model_name_key}...")
        results = yolo_model_instance(image, verbose=False, conf=0.25) # Use 0.25 as default confidence
        
        detections_list = []
        for r in results:
            for *xyxy, conf, cls in r.boxes.data.tolist():
                label_idx = int(cls)
                # Use the correct class names for the specific YOLO model
                label = YOLO_CLASS_NAMES[model_name_key][label_idx] if label_idx < len(YOLO_CLASS_NAMES[model_name_key]) else f"unknown_class_{label_idx}"
                detections_list.append({
                    "box": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                    "score": float(conf),
                    "label": label
                })
        print(f"YOLO '{model_name_key}' detections: {len(detections_list)} objects found.")
        return detections_list
    except Exception as e:
        print(f"ERROR during YOLO '{model_name_key}' prediction: {e}")
        print(f"Detailed YOLO prediction error for {model_name_key}: {e.__class__.__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"YOLO '{model_name_key}' prediction failed: {e}")


def predict_cnn_resnet50(image: Image.Image, score_threshold: float = 0.2) :
    """
    Performs inference using the Faster R-CNN ResNet50 model.
    Returns a list of dictionaries, each representing a detected object.
    Each dict contains 'box' (xmin, ymin, xmax, ymax), 'score', and 'label'.
    """
    model = cnn_resnet50_model # Use the globally loaded model
    if model is None:
        raise HTTPException(status_code=500, detail=f"Faster R-CNN (ResNet50) model not loaded. Check server logs for details.")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Performing CNN (ResNet50) prediction...")

        img_tensor = F.to_tensor(image)
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        img_tensor = img_tensor.to(device)
        img_list = [img_tensor]

        with torch.no_grad():
            predictions = model(img_list)

        detections_list = []
        if predictions:
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            for i in range(len(scores)):
                if scores[i] > score_threshold:
                    label_idx = int(labels[i])
                    label = CNN_CLASS_NAMES[label_idx] if label_idx < len(CNN_CLASS_NAMES) else f"unknown_class_{label_idx}"
                    detections_list.append({
                        "box": boxes[i].tolist(),
                        "score": float(scores[i]),
                        "label": label
                    })
        print(f"CNN (ResNet50) detections: {len(detections_list)} objects found.")
        return detections_list
    except Exception as e:
        print(f"ERROR during CNN prediction: {e}")
        print(f"Detailed CNN prediction error: {e.__class__.__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {e}")