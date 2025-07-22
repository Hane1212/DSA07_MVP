# utils/api.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from PIL import Image
import io
import torch
import pickle
import numpy as np
import cv2 # For drawing bounding boxes and image manipulation
from typing import List, Dict, Union, Optional
import csv # For reading the ground truth file
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, 
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional as F # For image preprocessing

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="Fruit Detection API",
    description="API for detecting fruits using YOLO and CNN models.",
    version="1.0.0"
)

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

def predict_yolo(yolo_model_instance: YOLO, model_name_key: str, image: Image.Image) -> List[Dict[str, Union[str, float, List[float]]]]:
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


def predict_cnn_resnet50(image: Image.Image, score_threshold: float = 0.2) -> List[Dict[str, Union[str, float, List[float]]]]:
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

# --- API Endpoints ---

@fastapi_app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_names: List[str] = Form(...), # Now accepts a list of model names
    image_filename: str = Form(...) # Get filename from Streamlit
):
    """
    Endpoint to receive an image and perform fruit detection using specified models.
    `model_names` can be a single model or a list for comparison.
    Example: model_names=["YOLOv10m"], model_names=["FasterRCNN"], model_names=["YOLOv10m", "FasterRCNN"]
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = {}

    # Get ground truth count for the specific image
    ground_truth_count = ground_truth_counts.get(image_filename, None)
    results["ground_truth_count"] = ground_truth_count
    if ground_truth_count is None:
        print(f"WARNING: No ground truth found for image: {image_filename}")

    for model_name in model_names:
        if model_name == "YOLOv10m":
            try:
                yolo_model_instance = yolo_models.get("YOLOv10m")
                if yolo_model_instance:
                    yolo_res = predict_yolo(yolo_model_instance, "YOLOv10m", image)
                    results["YOLOv10m"] = yolo_res
                else:
                    results["YOLOv10m_error"] = "YOLOv10m model not loaded."
            except HTTPException as e:
                results["YOLOv10m_error"] = str(e.detail)
            except Exception as e:
                results["YOLOv10m_error"] = f"YOLOv10m prediction failed: {e}"
                print(f"YOLOv10m prediction error: {e}")
        elif model_name == "YOLOv9c":
            try:
                yolo_model_instance = yolo_models.get("YOLOv9c")
                if yolo_model_instance:
                    yolo_res = predict_yolo(yolo_model_instance, "YOLOv9c", image)
                    results["YOLOv9c"] = yolo_res
                else:
                    results["YOLOv9c_error"] = "YOLOv9c model not loaded."
            except HTTPException as e:
                results["YOLOv9c_error"] = str(e.detail)
            except Exception as e:
                results["YOLOv9c_error"] = f"YOLOv9c prediction failed: {e}"
                print(f"YOLOv9c prediction error: {e}")
        elif model_name == "YOLOv10l": # Added YOLOv10l
            try:
                yolo_model_instance = yolo_models.get("YOLOv10l")
                if yolo_model_instance:
                    yolo_res = predict_yolo(yolo_model_instance, "YOLOv10l", image)
                    results["YOLOv10l"] = yolo_res
                else:
                    results["YOLOv10l_error"] = "YOLOv10l model not loaded."
            except HTTPException as e:
                results["YOLOv10l_error"] = str(e.detail)
            except Exception as e:
                results["YOLOv10l_error"] = f"YOLOv10l prediction failed: {e}"
                print(f"YOLOv10l prediction error: {e}")
        elif model_name == "FasterRCNN":
            try:
                cnn_res = predict_cnn_resnet50(image) 
                results["FasterRCNN"] = cnn_res
            except HTTPException as e:
                results["FasterRCNN_error"] = str(e.detail)
            except Exception as e:
                results["FasterRCNN_error"] = f"FasterRCNN prediction failed: {e}"
                print(f"FasterRCNN prediction error: {e}")
        else:
            print(f"Unknown model requested: {model_name}")

    if not results:
        raise HTTPException(status_code=400, detail="No valid model type selected or models failed to load/predict.")

    return results

# --- Startup Event ---
@fastapi_app.on_event("startup")
async def startup_event():
    """Load models and ground truth on startup."""
    print("Loading models and ground truth on startup...")
    load_yolo_model("YOLOv10m", "models/yolov10m.pt")
    load_yolo_model("YOLOv9c", "models/yolov9c.pt")
    load_yolo_model("YOLOv10l", "models/yolov10l.pt") # Added YOLOv10l model loading
    load_cnn_resnet50_model()
    load_ground_truth() # Load ground truth data
    print("Models and ground truth loading attempt completed.")
