# utils/detection_utils.py

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import pandas as pd
from fpdf import FPDF

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

        pdf.image(img_buffer_combined, x=pdf.l_margin, y=pdf.get_y(), w=pdf_image_width, h=pdf_image_height, type="PNG")
        pdf.ln(pdf_image_height + 5)

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

    return bytes(pdf.output(dest='S'))
