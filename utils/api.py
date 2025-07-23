# utils/api.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
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

# Import the simulated authentication utilities
from utils import auth_utils
import os
from twilio.rest import Client
from fastapi import APIRouter

router = APIRouter()

# Define request model
class SMSRequest(BaseModel):
    username: str
    file_name: str
    yield_kg: float
    phone_number: str


# Initialize FastAPI app
fastapi_app = FastAPI(
    title="Fruit Detection API",
    description="API for detecting fruits using YOLO and CNN models with authentication.",
    version="1.0.0"
)

# --- Pydantic Models for Authentication ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str # "admin", "annotator", "viewer"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str

class UserPayload(BaseModel):
    username: str
    role: str

# --- OAuth2PasswordBearer for token extraction from headers ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- Global variables for models and ground truth ---
yolo_models = {} 
cnn_resnet50_model = None
ground_truth_counts = {} 

YOLO_CLASS_NAMES = {
    "YOLOv10m": [],
    "YOLOv9c": [],
    "YOLOv10l": [] 
} 

CNN_CLASS_NAMES = [
    "__background__", "fruit" 
] 

# --- Ground Truth Loading Function ---
def load_ground_truth():
    """Loads ground truth counts from ground_truth.txt."""
    global ground_truth_counts
    print("Loading ground_truth.txt...")
    try:
        with open('ground_truth.txt', 'r') as f: 
            reader = csv.reader(f)
            header = next(reader) 
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
            yolo_instance.eval() 
            
            if hasattr(yolo_instance, 'names') and yolo_instance.names:
                YOLO_CLASS_NAMES[model_name] = yolo_instance.names 
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

            cnn_expected_num_classes = 2 
            
            model_path = 'models/fasterrcnn_resNet50_fpn.pth' 
            cnn_instance = fasterrcnn_resnet50_fpn(weights=None, num_classes=cnn_expected_num_classes)

            print(f"Initializing Faster R-CNN (ResNet50) with num_classes={cnn_expected_num_classes}")
            
            print(f"Loading state_dict from '{model_path}' with map_location=torch.device('cpu')")
            state_dict = torch.load(
                model_path, 
                map_location=torch.device('cpu')
            )
            
            cnn_instance.load_state_dict(state_dict)
            cnn_instance.to(device) 
            cnn_instance.eval() 
            cnn_resnet50_model = cnn_instance
            print(f"Faster R-CNN (ResNet50) model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Faster R-CNN (ResNet50) model: {e}")
            print(f"Detailed CNN loading error: {e.__class__.__name__}: {e}")
            cnn_resnet50_model = None 
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
        results = yolo_model_instance(image, verbose=False, conf=0.25) 
        
        detections_list = []
        for r in results:
            for *xyxy, conf, cls in r.boxes.data.tolist():
                label_idx = int(cls)
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
    model = cnn_resnet50_model 
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

# --- Authentication Dependencies ---
def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPayload:
    """Authenticates user based on token and returns user payload."""
    user = auth_utils.get_user_from_token(token) # Simulate TinyAuth token validation
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserPayload(**user)

def get_admin_user(current_user: UserPayload = Depends(get_current_user)):
    """Ensures the current user has 'admin' role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions: Admin role required",
        )
    return current_user

def get_annotator_user(current_user: UserPayload = Depends(get_current_user)):
    """Ensures the current user has 'annotator' or 'admin' role."""
    if current_user.role not in ["annotator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions: Annotator or Admin role required",
        )
    return current_user

def get_viewer_user(current_user: UserPayload = Depends(get_current_user)):
    """Ensures the current user has 'viewer', 'annotator', or 'admin' role."""
    if current_user.role not in ["viewer", "annotator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions: Viewer, Annotator, or Admin role required",
        )
    return current_user

# --- API Endpoints ---

# Authentication Endpoints
@fastapi_app.post("/auth/register", response_model=TokenResponse)
async def register(user: UserCreate):
    """Register a new user with a specified role."""
    registered_user = auth_utils.register_user(user.username, user.password, user.role)
    if not registered_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    return registered_user

@fastapi_app.post("/auth/login", response_model=TokenResponse)
async def login(user_credentials: UserLogin):
    """Log in a user and return an access token."""
    user_data = auth_utils.login_user(user_credentials.username, user_credentials.password)
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_data

# Protected Prediction Endpoint (Accessible by all logged-in users)
@fastapi_app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_names: List[str] = Form(...), 
    image_filename: str = Form(...),
    current_user: UserPayload = Depends(get_viewer_user) # All roles can predict/compare
):
    """
    Endpoint to receive an image and perform fruit detection using specified models.
    Requires authentication. Accessible by Viewer, Annotator, Admin roles.
    """
    print(f"User '{current_user.username}' ({current_user.role}) is requesting prediction.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = {}

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
                results["Yolo9c_error"] = str(e.detail)
            except Exception as e:
                results["YOLOv9c_error"] = f"YOLOv9c prediction failed: {e}"
                print(f"YOLOv9c prediction error: {e}")
        elif model_name == "YOLOv10l": 
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

# Placeholder Protected Endpoints for different roles
@fastapi_app.post("/admin/train", tags=["Admin"], summary="Train models (Admin only)")
async def train_models(current_user: UserPayload = Depends(get_admin_user)):
    """Allows admin users to trigger model training."""
    return {"message": f"User '{current_user.username}' (Admin) is training models. (Placeholder)"}

@fastapi_app.delete("/admin/records/{record_id}", tags=["Admin"], summary="Delete records (Admin only)")
async def delete_record(record_id: str, current_user: UserPayload = Depends(get_admin_user)):
    """Allows admin users to delete records."""
    return {"message": f"User '{current_user.username}' (Admin) is deleting record {record_id}. (Placeholder)"}

@fastapi_app.post("/annotator/correct_prediction", tags=["Annotator"], summary="Correct predictions (Annotator/Admin only)")
async def correct_prediction(current_user: UserPayload = Depends(get_annotator_user)):
    """Allows annotator or admin users to correct predictions."""
    return {"message": f"User '{current_user.username}' ({current_user.role}) is correcting predictions. (Placeholder)"}


# --- Startup Event ---
@fastapi_app.on_event("startup")
async def startup_event():
    """Load models and ground truth on startup."""
    print("Loading models and ground truth on startup...")
    load_yolo_model("YOLOv10m", "models/yolov10m.pt")
    load_yolo_model("YOLOv9c", "models/yolov9c.pt")
    load_yolo_model("YOLOv10l", "models/yolov10l.pt") 
    load_cnn_resnet50_model()
    load_ground_truth() 
    print("Models and ground truth loading attempt completed.")

@router.post("/send_sms")
async def send_sms(sms: SMSRequest):
    try:
        # Load credentials from environment or secrets
        account_sid = os.getenv("TWILIO_ACCOUNT_SID", "your_sid_here")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "your_token_here")
        from_number = os.getenv("TWILIO_FROM_NUMBER", "+1234567890")  # your Twilio number

        client = Client(account_sid, auth_token)

        message_body = (
            f"Hello {sms.username}, the estimated yield for {sms.file_name} is {sms.yield_kg:.2f} kg."
        )

        message = client.messages.create(
            body=message_body,
            from_=from_number,
            to=sms.phone_number
        )

        return {"status": "success", "message_sid": message.sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send SMS: {e}")
