# pages/_Compare.py

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import pandas as pd
import altair as alt
import os

# Import utility functions
from utils.detection_utils import draw_boxes_on_image, create_prediction_pdf

# IMPORTANT: Use the service name 'fastapi' as defined in docker-compose.yml
# for inter-container communication.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000") 

# --- Session State Initialization for this page ---
# These variables are specific to the _Compare.py page
if 'compare_predictions' not in st.session_state:
    st.session_state.compare_predictions = None
if 'compare_original_image_bytes' not in st.session_state:
    st.session_state.compare_original_image_bytes = None
if 'compare_current_image_filename' not in st.session_state:
    st.session_state.compare_current_image_filename = None
if 'compare_combined_img_for_pdf' not in st.session_state:
    st.session_state.compare_combined_img_for_pdf = None
if 'compare_metrics_df_for_pdf' not in st.session_state:
    st.session_state.compare_metrics_df_for_pdf = None
if 'compare_page_view' not in st.session_state:
    st.session_state.compare_page_view = 'selection' # 'selection' or 'dashboard'

def render_compare_selection_page():
    st.title("⚖️ Model Comparison")
    st.markdown("Select two models and an image to compare their detection performance.")

    # Model selection dropdowns
    available_models = ["YOLOv10m", "YOLOv9c", "YOLOv10l", "FasterRCNN"] # Added YOLOv10l
    
    col1, col2 = st.columns(2)
    with col1:
        model1_name = st.selectbox("Select Model 1", available_models, key="model1_select")
    with col2:
        # Ensure model2 is different from model1 if possible
        filtered_models_for_m2 = [m for m in available_models if m != model1_name]
        if not filtered_models_for_m2:
            model2_name = model1_name # Fallback if only one model type exists
        else:
            default_index_m2 = 0
            # Try to set a different default for Model 2 if possible
            if model1_name == "YOLOv10m" and "YOLOv9c" in filtered_models_for_m2:
                default_index_m2 = filtered_models_for_m2.index("YOLOv9c")
            elif model1_name == "FasterRCNN" and "YOLOv10m" in filtered_models_for_m2:
                default_index_m2 = filtered_models_for_m2.index("YOLOv10m")
            elif model1_name == "YOLOv9c" and "YOLOv10l" in filtered_models_for_m2:
                default_index_m2 = filtered_models_for_m2.index("YOLOv10l")
            
            model2_name = st.selectbox("Select Model 2", filtered_models_for_m2, index=default_index_m2, key="model2_select")

    if model1_name == model2_name:
        st.warning("Please select two different models for comparison.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="compare_uploader")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(uploaded_file, caption="Uploaded Image for Comparison", use_column_width=True)
        st.write("")

        # This button triggers the prediction and stores results in session state
        if st.button("Compare Models"):
            with st.spinner(f"Comparing {model1_name} and {model2_name} predictions..."):
                try:
                    data = {
                        "model_names": [model1_name, model2_name],
                        "image_filename": uploaded_file.name
                    }
                    files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    
                    predict_url = f"{FASTAPI_URL}/predict"
                    
                    response = requests.post(predict_url, data=data, files=files)
                    response.raise_for_status() 
                    
                    predictions = response.json()
                    st.session_state.compare_predictions = predictions # Store for dashboard
                    st.session_state.compare_original_image_bytes = image_bytes
                    st.session_state.compare_current_image_filename = uploaded_file.name

                    st.subheader("Comparison Results:")
                    
                    combined_img = original_image.copy()
                    
                    # Process Model 1 results
                    model1_detections = predictions.get(model1_name, [])
                    if isinstance(model1_detections, list):
                        st.markdown(f"### {model1_name} Results:")
                        if model1_detections:
                            st.write(f"Detected **{len(model1_detections)}** objects.")
                            img_with_model1_boxes = original_image.copy()
                            img_with_model1_boxes = draw_boxes_on_image(img_with_model1_boxes, model1_detections, model1_name)
                            st.image(img_with_model1_boxes, caption=f"{model1_name} Detections", use_column_width=True)
                            combined_img = draw_boxes_on_image(combined_img, model1_detections, model1_name)
                        else:
                            st.write(f"No objects detected by {model1_name} model.")
                    elif f"{model1_name}_error" in predictions:
                        st.error(f"{model1_name} Model Error: {predictions[f'{model1_name}_error']}")

                    st.markdown("---")

                    # Process Model 2 results
                    model2_detections = predictions.get(model2_name, [])
                    if isinstance(model2_detections, list):
                        st.markdown(f"### {model2_name} Results:")
                        if model2_detections:
                            st.write(f"Detected **{len(model2_detections)}** objects.")
                            img_with_model2_boxes = original_image.copy()
                            img_with_model2_boxes = draw_boxes_on_image(img_with_model2_boxes, model2_detections, model2_name)
                            st.image(img_with_model2_boxes, caption=f"{model2_name} Detections", use_column_width=True)
                            combined_img = draw_boxes_on_image(combined_img, model2_detections, model2_name)
                        else:
                            st.write(f"No objects detected by {model2_name} model.")
                    elif f"{model2_name}_error" in predictions:
                        st.error(f"{model2_name} Model Error: {predictions[f'{model2_name}_error']}")

                    st.markdown("---")
                    st.subheader("Combined Detections")
                    st.image(combined_img, caption="Combined Detections (Model 1: Red, Model 2: Green/Blue)", use_column_width=True)
                    st.session_state.compare_combined_img_for_pdf = combined_img # Store for PDF

                    # Generate metrics for dashboard and PDF
                    ground_truth_count = predictions.get("ground_truth_count")
                    
                    # Calculate metrics for both models
                    metrics_data = {
                        "Metric": ["Number of Detections", "Average Confidence", "Simulated Precision*", "Simulated Recall*", "Simulated F1-Score*"],
                        model1_name: [
                            len(model1_detections), 
                            np.mean([d['score'] for d in model1_detections]) if model1_detections else 0.0, 
                            np.random.uniform(0.7, 0.95),
                            np.random.uniform(0.6, 0.90),
                            np.random.uniform(0.65, 0.92)
                        ],
                        model2_name: [
                            len(model2_detections), 
                            np.mean([d['score'] for d in model2_detections]) if model2_detections else 0.0, 
                            np.random.uniform(0.5, 0.85),
                            np.random.uniform(0.4, 0.80),
                            np.random.uniform(0.45, 0.82)
                        ]
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    st.session_state.compare_metrics_df_for_pdf = df_metrics # Store for dashboard and PDF

                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to FastAPI server at {FASTAPI_URL}. Please ensure the server is running.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred during prediction: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            # After prediction, if successful, display the dashboard button
            # This reruns the script, and the dashboard button will be rendered outside this 'if' block
            st.session_state.compare_page_view = 'dashboard'
            st.rerun() # Trigger a rerun to show the dashboard
    else:
        st.info("Please upload an image to compare models.")

    # This button is now outside the "Compare Models" if block,
    # and only appears if there are predictions to show in the dashboard.
    if st.session_state.compare_predictions and st.session_state.compare_page_view == 'selection':
        st.markdown("---")
        st.subheader("Actions")
        col_dash, col_pdf = st.columns(2)
        with col_dash:
            if st.button("Go to Comparison Dashboard (View Previous Results)"):
                st.session_state.compare_page_view = 'dashboard'
                st.rerun()
        with col_pdf:
            if st.session_state.compare_metrics_df_for_pdf is not None:
                pdf_bytes = create_prediction_pdf(
                    st.session_state.compare_original_image_bytes,
                    st.session_state.compare_current_image_filename,
                    {k: v for k, v in st.session_state.compare_predictions.items() if k not in ["ground_truth_count", "YOLOv10m_error", "YOLOv9c_error", "YOLOv10l_error", "FasterRCNN_error"]},
                    st.session_state.compare_predictions.get("ground_truth_count"), 
                    st.session_state.compare_metrics_df_for_pdf,
                    st.session_state.compare_combined_img_for_pdf
                )
                st.download_button(
                    label="Download Comparison Report as PDF",
                    data=pdf_bytes,
                    file_name=f"{st.session_state.compare_current_image_filename}_comparison_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("Run a comparison first to enable PDF download.")


def render_comparison_dashboard_page():
    st.title("📊 Model Comparison Dashboard")
    st.markdown("---")

    # Go back button
    if st.button("← Back to Model Comparison Selection"):
        st.session_state.compare_page_view = 'selection'
        st.rerun() 

    st.markdown("---")

    if st.session_state.compare_predictions:
        predictions = st.session_state.compare_predictions
        ground_truth_count = predictions.get("ground_truth_count")
        
        # Get the names of the models that were compared
        compared_model_names = [k for k in predictions.keys() if k not in ["ground_truth_count", "YOLOv10m_error", "YOLOv9c_error", "YOLOv10l_error", "FasterRCNN_error"]] 
        
        if len(compared_model_names) == 2:
            model1_name = compared_model_names[0]
            model2_name = compared_model_names[1]
            model1_detections = predictions.get(model1_name, [])
            model2_detections = predictions.get(model2_name, [])
        else:
            st.warning("Dashboard is designed for comparing two models. Please run a comparison first.")
            return

        # Display combined image if available
        if st.session_state.compare_combined_img_for_pdf:
            st.image(st.session_state.compare_combined_img_for_pdf, caption="Combined Detections", use_container_width=True)
            st.markdown("---")

        st.markdown("### Detailed Comparison")
        model1_labels_set = {det['label'] for det in model1_detections}
        model2_labels_set = {det['label'] for det in model2_detections} 

        st.markdown(f"**{model1_name} detected {len(model1_detections)} objects.**")
        st.markdown(f"**{model2_name} detected {len(model2_detections)} objects.**")

        st.markdown("#### Label-Based Comparison:")
        if model1_labels_set and model2_labels_set:
            common_labels = model1_labels_set.intersection(model2_labels_set)
            model1_only = model1_labels_set.difference(model2_labels_set)
            model2_only = model2_labels_set.difference(model1_labels_set)

            if common_labels:
                st.success(f"**Common labels detected by both:** {', '.join(common_labels)}")
            else:
                st.info("No common labels detected by both models.")
            if model1_only:
                st.info(f"**Labels detected only by {model1_name}:** {', '.join(model1_only)}")
            if model2_only:
                st.info(f"**Labels detected only by {model2_name}:** {', '.join(model2_only)}")
        elif model1_labels_set:
            st.info(f"{model1_name} detected: {', '.join(model1_labels_set)}. {model2_name} did not detect any objects or had an error.")
        elif model2_labels_set:
            st.info(f"{model2_name} detected: {', '.join(model2_labels_set)}. {model1_name} did not detect any objects or had an error.")
        else:
            st.info("No clear comparison possible due to errors or no detections from either model.")

        st.markdown("#### Qualitative Performance Assessment:")
        st.write("Models show varying detection patterns. Further analysis with diverse images would be beneficial.")
        
        st.markdown(
            "**General Improvement Suggestion:** Consider collecting more diverse images, especially edge cases (e.g., partially hidden fruits, varying lighting conditions) to further improve both models' robustness."
        )

        st.markdown("---")
        st.markdown("## Model Performance Dashboard")
        st.info(
            "**Note:** Precision, Recall, and F1-Score for object detection typically require ground truth bounding box annotations. Here, we provide metrics based on **count comparison** and simulated values for illustration. For a full evaluation, a dedicated dataset with detailed annotations is needed."
        )

        if ground_truth_count is not None:
            st.markdown(f"**Ground Truth Count for this image:** {ground_truth_count}")
            model1_count = len(model1_detections)
            model2_count = len(model2_detections)
            
            model1_count_accuracy = 1.0 - (abs(model1_count - ground_truth_count) / max(ground_truth_count, 1))
            model2_count_accuracy = 1.0 - (abs(model2_count - ground_truth_count) / max(ground_truth_count, 1))
            
            st.markdown(f"**{model1_name} Count Accuracy:** {model1_count_accuracy:.2f}")
            st.markdown(f"**{model2_name} Count Accuracy:** {model2_count_accuracy:.2f}")
        else:
            st.warning("Ground Truth Count not available for this image.")

        if st.session_state.compare_metrics_df_for_pdf is not None:
            df_metrics = st.session_state.compare_metrics_df_for_pdf
            st.table(df_metrics.style.format({
                "Average Confidence": "{:.2f}",
                "Simulated Precision*": "{:.2f}",
                "Simulated Recall*": "{:.2f}",
                "Simulated F1-Score*": "{:.2f}"
            }))

            st.markdown("#### Visual Comparison of Key Metrics:")

            # Prepare data for charts dynamically based on compared models
            chart_data_counts = pd.DataFrame({
                "Model": [model1_name, model2_name],
                "Count": [len(model1_detections), len(model2_detections)]
            })

            chart_detections = alt.Chart(chart_data_counts).mark_bar().encode(
                x=alt.X('Model:N', axis=None), 
                y=alt.Y('Count:Q', title='Number of Detections'),
                color=alt.Color('Model:N', legend=alt.Legend(title="Model")),
                tooltip=['Model', 'Count']
            ).properties(
                title='Number of Detections'
            )
            st.altair_chart(chart_detections, use_container_width=True)

            model1_avg_conf = np.mean([d['score'] for d in model1_detections]) if model1_detections else 0.0
            model2_avg_conf = np.mean([d['score'] for d in model2_detections]) if model2_detections else 0.0
            chart_data_conf = pd.DataFrame({
                "Model": [model1_name, model2_name],
                "Average Confidence": [model1_avg_conf, model2_avg_conf]
            })
            chart_confidence = alt.Chart(chart_data_conf).mark_bar().encode(
                x=alt.X('Model:N', axis=None), 
                y=alt.Y('Average Confidence:Q', title='Average Confidence', scale=alt.Scale(domain=[0, 1])), 
                color=alt.Color('Model:N', legend=alt.Legend(title="Model")),
                tooltip=['Model', alt.Tooltip('Average Confidence', format='.2f')]
            ).properties(
                title='Average Confidence Score'
            )
            st.altair_chart(chart_confidence, use_container_width=True)
        else:
            st.info("No metrics data available. Please run a comparison on the 'Compare Models' tab first.")
    else:
        st.info("No predictions available. Please upload an image and run comparison on the 'Compare Models' tab first.")

    # PDF Download button for Dashboard Page
    st.markdown("---")
    st.subheader("Download Prediction Report")
    if st.session_state.compare_predictions and st.session_state.compare_metrics_df_for_pdf is not None:
        pdf_bytes = create_prediction_pdf(
            st.session_state.compare_original_image_bytes,
            st.session_state.compare_current_image_filename,
            {k: v for k, v in st.session_state.compare_predictions.items() if k not in ["ground_truth_count", "YOLOv10m_error", "YOLOv9c_error", "YOLOv10l_error", "FasterRCNN_error"]}, # Filter out non-detection keys
            st.session_state.compare_predictions.get("ground_truth_count"), 
            st.session_state.compare_metrics_df_for_pdf,
            st.session_state.compare_combined_img_for_pdf
        )
        st.download_button(
            label="Download Comparison Report as PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.compare_current_image_filename}_comparison_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Please run a comparison on the 'Compare Models' tab to enable PDF download.")


# --- Main Logic for _Compare.py ---
if st.session_state.compare_page_view == 'selection':
    render_compare_selection_page()
elif st.session_state.compare_page_view == 'dashboard':
    render_comparison_dashboard_page()
