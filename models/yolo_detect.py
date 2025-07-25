from ultralytics import YOLO

# Load main model (large)
model_large = YOLO("models/yolov10l.pt")

# Lazy-load small model for streaming
model_small = None

def predict_fruit_count(image_path_or_frame, return_results=False, fast=False):
    """
    fast=True  -> use small model (yolov10s.pt)
    fast=False -> use large model (yolov10l.pt)
    """
    global model_small

    model_to_use = model_large
    if fast:
        if model_small is None:
            model_small = YOLO("models/yolov10s.pt")
        model_to_use = model_small

    results = model_to_use.predict(image_path_or_frame, stream=False)
    if return_results:
        return results
    return len(results[0].boxes)
