# DSA07_MVP
# 🍎 Smart Fruit Detection & Yield Estimation Platform

This project is a web-based application that uses deep learning and real-time data to detect fruits from orchard images, estimate yield, and calculate potential revenue based on weather and market prices. It supports multiple detection models (YOLOv9, YOLOv10, Faster R-CNN) and includes interactive editing, historical tracking, and chatbot integration.

---

## Key Features From Main Screen
### Detection tab
1. Upload images of fruit trees (e.g., apple orchards)
2. Detect fruit using trained object detection models: 
    + YOLOv10m 
    + YOLOv9
    + fasterRCNNmobile
    + fasterRCNNresNet50
3. Edit brightness, crop image
4. View detection result
5. Save results to a prosgesSQL database
### AgriBot Chat
Allow User ask chatbot about agriculture field, using API from **gemini-2.5-pro**

### History/log
- Display all information about images which was detectioned from App and saved to database before.
- Allow Filter by image name, the minium counter fruit or load all image

### Estimate
- Based on the number of counting from the Detection tab, User can get the estimage report with weight, price, classficiation about size.
- Fetch real-time weather (OpenWeather API)
- Fetch real-time crop prices (Agmarknet API - India)
- Estimate total yield (kg) and market revenue (₹)

### Enhanced Analysis
---
## Compare page
The Compare Model page lets users view the performance of different YOLO versions (v9, v10m,v10l,RCNN) side by side. It shows key metrics like accuracy, precision, and  recall, making it easier to choose the right model for specific needs. Users can upload an image and instantly compare how each model detects fruits, including confidence scores and bounding boxes.  After comparison, users can download the results for both images with detections and the summary report.

---

**Models Trained:**
- YOLOv9
- YOLOv10
- Faster R-CNN (ResNet, MobileNet)
- Trained on the [MinneApple dataset](https://github.com/DocF/MinneApple)

---
## ProsgessSQL db
**table: ImageDetection**

The app stores all detection and correction metadata in a local or cloud-connected database. Each record corresponds to one processed image.

| Column                | Type      | Description                                      |
|-----------------------|-----------|--------------------------------------------------|
| `image_id`            | String    | Unique ID or filename of the image              |
| `image_path`          | String    | Path to saved annotated image                   |
| `model_name`          | String    | Detection model used (e.g., YOLOv10m)           |
| `detection_time`      | Float     | Time taken for model inference (in seconds)     |
| `confidence`          | Float     | Average model confidence score                  |
| `model_predictions`   | JSON      | Raw bounding boxes predicted by the model       |
| `user_annotations`    | JSON      | User-corrected annotations (optional)           |
| `num_objects_model`   | Integer   | Number of fruits detected by the model          |
| `num_objects_corrected` | Integer | Manually corrected fruit count                  |
| `annotated_by_user`   | Boolean   | Whether the result was manually corrected       |
| `annotator_id`        | String    | User ID of annotator (if any)                   |
| `annotation_date`     | DateTime  | Timestamp when annotation was saved             |

> All detections can be browsed in the "History / Logs" tab in the app.


---

## ⚙️ Installation Instructions

> ✅ Recommended: Python 3.9

1. **Clone the repository**
```bash
git clone https://github.com/Hane1212/DSA07_MVP.git
cd fruit-detection-estimation
```
2. **Add model to models folder**

3. **run docker file**  
```bash
docker compose build
docker compose up -d
```
4. **Access to the application**  
```bash
http://0.0.0.0:8501/
```


