import requests
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from fpdf import FPDF

LOCAL_HOST = "http://fastapi:8000/"  # Optional if using FastAPI backend

# ---------------- SAVE TO CSV ----------------

def save_to_csv(result, model_name, file_path="history.csv"):
    row = {
        "image_id": result["image_id"],
        "image_path": result["image_path"],
        "boxed_path": result.get("boxed_path", ""),
        "predicted_count": result["fruit_count"],
        "corrected_count": result.get("corrected_count", result["fruit_count"]),
        "model_used": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.drop_duplicates(subset=["image_id", "predicted_count", "corrected_count", "model_used"], keep="last", inplace=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(file_path, index=False)
    return {"message": "✅ Saved to CSV"}

# ---------------- SHOW HISTORY ----------------

def show_detection_history(csv_path="history.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.empty:
            st.info("No detection history available yet.")
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("History file not found. Try saving a detection first.")

# ---------------- PDF GENERATION ----------------

def generate_pdf_from_csv(csv_path="history.csv", output_path="correction_report.pdf"):
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Fruit Counting Report", ln=True, align="C")
    pdf.ln(10)

    for _, row in df.iterrows():
        image_path = str(row.get("image_path", "")).strip()
        boxed_path = str(row.get("boxed_path", "")).strip()
        image_id = str(row.get("image_id", "Unknown"))
        predicted = str(row.get("predicted_count", "N/A"))
        corrected = str(row.get("corrected_count", "N/A"))
        model = str(row.get("model_used", "N/A"))
        timestamp = str(row.get("timestamp", "N/A"))

        # Cropped Image
        if image_path and image_path.lower() != "nan" and os.path.exists(image_path):
            try:
                pdf.image(image_path, x=10, w=60)
            except Exception as e:
                pdf.cell(0, 10, f"[Image Error: {e}]", ln=True)

        # YOLO Output with Bounding Boxes
        if boxed_path and boxed_path.lower() != "nan" and os.path.exists(boxed_path):
            try:
                pdf.image(boxed_path, x=75, w=60)
            except Exception as e:
                pdf.cell(0, 10, f"[Boxed Image Error: {e}]", ln=True)

        # Text Section
        pdf.set_font("Arial", size=10)
        pdf.ln(3)
        pdf.multi_cell(0, 10, txt=(
            f"Image: {image_id}\n"
            f"Predicted Count: {predicted}\n"
            f"Corrected Count: {corrected}\n"
            f"Model: {model}\n"
            f"Timestamp: {timestamp}"
        ))
        pdf.ln(10)

    pdf.output(output_path)
    return output_path

# ---------------- BACKEND API (Optional) ----------------

def run_detection_api(file, model_name):
    files = {"file": (file.name, file.getvalue(), "image/png")}
    data = {"model_name": model_name}
    response = requests.post(f"{LOCAL_HOST}/predict/", files=files, data=data)
    response.raise_for_status()
    return response.json()

def save_detection_api(result, model_name):
    payload = {
        "image_id": result["image_id"],
        "image_path": result["image_path"],
        "model_name": model_name,
        "detection_time": result.get("detection_time", datetime.now().isoformat()),
        "confidence": result.get("confidence", 0.0),
        "model_predictions": result.get("model_predictions", []),
        "user_annotations": [],
        "num_objects_model": result["fruit_count"],
        "num_objects_corrected": result.get("corrected_count", result["fruit_count"]),
        "annotated_by_user": True,
        "annotator_id": None
    }

    response = requests.post(f"{LOCAL_HOST}/save", json=payload)
    response.raise_for_status()
    return response.json()
# Temp edit to force Git update
