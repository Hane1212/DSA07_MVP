from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Initialize FastAPI app
fastapi_app = FastAPI()

# Allow CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
yolo_model = YOLO("./models/yolov10m.pt")

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
    output_path = "./output/images/"+filename
    cv2.imwrite(output_path, image)

    detection_time = round(time.time() - start, 2)
    return JSONResponse(content={
        "fruit_count": fruit_count,
        "confidence": float(boxes[0].conf[0]) if fruit_count > 0 else 0,
        "detection_time": detection_time
    })
