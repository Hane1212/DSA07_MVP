from ultralytics import YOLO
import os

model = YOLO("models/yolov10l.pt")

def predict_fruit_count(image_path, return_results=False):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(image_path)
    if return_results:
        return results
    return len(results[0].boxes)
