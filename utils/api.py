from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import List, Optional

from database.config import get_db, Base, async_engine
from database.models import ImageDetection

import time
import cv2
import numpy as np
from ultralytics import YOLO
import os

fastapi_app = FastAPI()

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
yolo_model = YOLO("./models/yolov10m.pt")

@fastapi_app.on_event("startup")
async def on_startup():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ----------------------------
# PREDICT endpoint
# ----------------------------
@fastapi_app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    start = time.time()
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = yolo_model(image)[0]
    boxes = results.boxes
    fruit_count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    filename = file.filename
    output_path = f"./output/images/{filename}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

    detection_time = round(time.time() - start, 2)
    return JSONResponse(content={
        "fruit_count": fruit_count,
        "confidence": float(boxes[0].conf[0]) if fruit_count > 0 else 0,
        "detection_time": detection_time,
        "image_path": output_path,
        "image_id": filename
    })

# ----------------------------
# Detection DB endpoints
# ----------------------------
class DetectionIn(BaseModel):
    image_id: str
    image_path: str
    model_name: str
    detection_time: float
    confidence: float
    model_predictions: List[dict]
    user_annotations: Optional[List[dict]] = []
    num_objects_model: int
    num_objects_corrected: Optional[int] = 0
    annotated_by_user: bool = False
    annotator_id: Optional[str] = None
    fruit_type: Optional[str] = None
    expected_revenue: Optional[float] = None
    market_price: Optional[float] = None
    estimated_workload: Optional[float] = None

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
        annotator_id=data.annotator_id,
        fruit_type=data.fruit_type,
        expected_revenue=data.expected_revenue,
        market_price=data.market_price,
        estimated_workload=data.estimated_workload
    )
    db.add(new_record)
    await db.commit()
    print(f"✅ Saved to DB: {data.image_id}")
    return {"message": "Detection saved."}

# @fastapi_app.get("/detections/{image_id}")
# async def get_detection(image_id: str, db: AsyncSession = Depends(get_db)):
#     result = await db.execute(select(ImageDetection).where(ImageDetection.image_id == image_id))
#     detection = result.scalar_one_or_none()
#     if not detection:
#         raise HTTPException(status_code=404, detail="Image detection not found")
#     return detection


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
