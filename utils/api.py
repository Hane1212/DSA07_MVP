from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from database.config import get_db, Base, async_engine
from database.models import ImageDetection, DetectionIn, TokenResponse, UserCreate, UserLogin, UserPayload
import time
import cv2
import numpy as np
import io
from ultralytics import YOLO
from torchvision.transforms import functional as F
import torch
import os
from utils import utils, authen
import streamlit as st
from torchvision import transforms as T



fastapi_app = FastAPI()

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR = os.path.dirname(BASE_DIR) 

NUM_CLASSES = 2  # 1 class: apple + 1 background
models = {
    "YOLOv10m": YOLO("./models/yolov10m.pt"),
    "YOLOv9": YOLO(os.path.join("./models/yolov9c.pt")),
    "fasterRCNNmobile": utils.load_faster_rcnn_mobilenet(NUM_CLASSES, os.path.join(ROOT_DIR,"models/fasterrcnn_mobilenet_fpn.pth")),
    "fasterRCNNresNet50": utils.load_faster_rcnn_resnet50(NUM_CLASSES, os.path.join(ROOT_DIR,"models/fasterrcnn_resNet50_fpn.pth")),
}

def get_model(model_name: str):
    if model_name not in models:
        print("❌ Invalid model name:", model_name)
        return JSONResponse(status_code=400, content={"error": "Model not found."})
 

@fastapi_app.on_event("startup")
async def on_startup():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        """Load models and ground truth on startup."""
    print("Loading models and ground truth on startup...")
    utils.load_yolo_model("YOLOv10m", "models/yolov10m.pt")
    utils.load_yolo_model("YOLOv9c", "models/yolov9c.pt")
    utils.load_yolo_model("YOLOv10l", "models/yolov10l.pt") # Added YOLOv10l model loading
    utils.load_cnn_resnet50_model()
    utils.load_ground_truth() # Load ground truth data
    print("Models and ground truth loading attempt completed.")


"""
Author: Sree charan Lagudu 
Date: 2025-07-22
Description: (Huong TA update to call all models)
"""
@fastapi_app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    print("🚀 API Called")
    start = time.time()

    if model_name not in models:
        return JSONResponse(status_code=400, content={"error": f"Unknown model: {model_name}"})

    model = models[model_name]
    print(f"🧠 Selected model: {model_name}")

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    filename = file.filename
    print(f"📁 Received file: {filename}")

    detections = []
    fruit_count = 0
    annotated = image.copy()

    # --- YOLO ---
    if "yolo" in model_name.lower():
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(rgb_img)[0]  # Ultralytics YOLO result
        boxes = results.boxes

        fruit_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            label = results.names[class_id]

            detections.append({
                "label": label,
                "score": score,
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Faster R-CNN ---
    elif "fasterrcnn" in model_name.lower():
        print("🚀 API call FasterRCNN")
        model.eval()
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to("cpu")

        with torch.no_grad():
            outputs = model(image_tensor)[0]

        threshold = 0.5
        boxes = outputs['boxes']
        scores = outputs['scores']
        labels = outputs['labels']
        keep = scores > threshold

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        fruit_count = len(boxes)

        for i in range(fruit_count):
            x1, y1, x2, y2 = boxes[i].int().tolist()
            score = float(scores[i])
            label = str(labels[i].item())  # or use a mapping if available

            detections.append({
                "label": label,
                "score": score,
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        return JSONResponse(status_code=400, content={"error": f"Unsupported model type: {model_name}"})

    # --- Save annotated image ---
    output_path = f"./output/images/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)

    detection_time = round(time.time() - start, 2)
    avg_conf = float(np.mean([d["score"] for d in detections])) if detections else 0.0

    return JSONResponse(content={
        "fruit_count": fruit_count,
        "confidence": avg_conf,
        "detection_time": detection_time,
        "image_path": output_path,
        "image_id": filename,
        "detections": detections  # ✅ Important for estimation tab
    })

"""
Author: Huong TA
Date: 2025-07-20
Description: API to save data to DB
"""
@fastapi_app.post("/save")
async def save_detection(data: DetectionIn, db: AsyncSession = Depends(get_db)):
    new_record = ImageDetection(
        image_id=data.image_id,
        image_path=data.image_path,
        model_name=data.model_name,
        detection_time=data.detection_time,
        confidence=data.confidence,
        model_predictions=data.model_predictions,
        user_annotations=data.user_annotations,
        num_objects_model=data.num_objects_model,
        num_objects_corrected=data.num_objects_corrected,
        annotated_by_user=data.annotated_by_user,
        annotator_id=data.annotator_id
    )
    db.add(new_record)
    await db.commit()
    print(f"✅ Saved to DB: {data.image_id}")
    return {"message": "Detection saved."}

"""
Author: Huong TA
Date: 2025-07-21
Description: API to get data from DB
"""
@fastapi_app.get("/detections")
async def get_recent_detections(
    image_id: Optional[str] = None,
    min_count: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    query = select(ImageDetection).order_by(ImageDetection.annotation_date.desc()).limit(50)

    if image_id:
        query = query.where(ImageDetection.image_id.ilike(f"%{image_id}%"))

    if min_count:
        query = query.where(ImageDetection.num_objects_model >= min_count)

    result = await db.execute(query)
    detections = result.scalars().all()

    return [d.__dict__ for d in detections]

########################################## Authentication Endpoints ####################################################
# 
# Placeholder Protected Endpoints for different roles
# @fastapi_app.post("/admin/train", tags=["Admin"], summary="Train models (Admin only)")
# async def train_models(current_user: UserPayload = Depends(authen.get_admin_user)):
#     """Allows admin users to trigger model training."""
#     return {"message": f"User '{current_user.username}' (Admin) is training models. (Placeholder)"}

# @fastapi_app.delete("/admin/records/{record_id}", tags=["Admin"], summary="Delete records (Admin only)")
# async def delete_record(record_id: str, current_user: UserPayload = Depends(authen.get_admin_user)):
#     """Allows admin users to delete records."""
#     return {"message": f"User '{current_user.username}' (Admin) is deleting record {record_id}. (Placeholder)"}

# @fastapi_app.post("/annotator/correct_prediction", tags=["Annotator"], summary="Correct predictions (Annotator/Admin only)")
# async def correct_prediction(current_user: UserPayload = Depends(authen.get_annotator_user)):
#     """Allows annotator or admin users to correct predictions."""
#     return {"message": f"User '{current_user.username}' ({current_user.role}) is correcting predictions. (Placeholder)"}

# @fastapi_app.post("/auth/register", response_model=TokenResponse)
# async def register(user: UserCreate):
#     """Register a new user with a specified role."""
#     registered_user = auth_utils.register_user(user.username, user.password, user.role)
#     if not registered_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username already registered"
#         )
#     return registered_user

# @fastapi_app.post("/auth/login", response_model=TokenResponse)
# async def login(user_credentials: UserLogin):
#     """Log in a user and return an access token."""
#     user_data = auth_utils.login_user(user_credentials.username, user_credentials.password)
#     if not user_data:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return user_data
########################################## COMPARE API ##################################################################
# ----------------------------
# COMPARE PAGE - Bhagyasri Parupudi
# ----------------------------
# --- API Endpoints ---

@fastapi_app.post("/compare")
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
    ground_truth_count = utils.ground_truth_counts.get(image_filename, None)
    results["ground_truth_count"] = ground_truth_count
    if ground_truth_count is None:
        print(f"WARNING: No ground truth found for image: {image_filename}")

    for model_name in model_names:
        if model_name == "YOLOv10m":
            try:
                yolo_model_instance = utils.yolo_models.get("YOLOv10m")
                if yolo_model_instance:
                    yolo_res = utils.predict_yolo(yolo_model_instance, "YOLOv10m", image)
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
                yolo_model_instance = utils.yolo_models.get("YOLOv9c")
                if yolo_model_instance:
                    yolo_res = utils.predict_yolo(yolo_model_instance, "YOLOv9c", image)
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
                yolo_model_instance = utils.yolo_models.get("YOLOv10l")
                if yolo_model_instance:
                    yolo_res = utils.predict_yolo(yolo_model_instance, "YOLOv10l", image)
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
                cnn_res = utils.predict_cnn_resnet50(image) 
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
