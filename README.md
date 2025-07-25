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

## Compare page


---

**Models Trained:**
- YOLOv9
- YOLOv10
- Faster R-CNN (ResNet, MobileNet)
- Trained on the [MinneApple dataset](https://github.com/DocF/MinneApple)

---
## ProsgessSQL db
**table: ImageDetection**
| Column Name             | Data Type  | Description                           | Constraints / Notes                   |
| ----------------------- | ---------- | ------------------------------------- | ------------------------------------- |
| `id`                    | `Integer`  | Primary key                           | `primary_key=True`, `index=True`      |
| `image_id`              | `String`   | Image file name or ID                 | `nullable=False`                      |
| `image_path`            | `String`   | File path of the annotated image      | `nullable=False`                      |
| `model_name`            | `String`   | Detection model used                  | `nullable=False`                      |
| `detection_time`        | `Float`    | Time taken for detection (seconds)    |                                       |
| `confidence`            | `Float`    | Average model confidence              |                                       |
| `model_predictions`     | `JSON`     | Original bounding boxes from model    | Stored as raw prediction data         |
| `user_annotations`      | `JSON`     | User-corrected bounding boxes         | Stored if user edits results          |
| `num_objects_model`     | `Integer`  | Count of objects detected by model    |                                       |
| `num_objects_corrected` | `Integer`  | Manually corrected fruit count        |                                       |
| `annotated_by_user`     | `Boolean`  | Flag indicating if manually corrected | Default: `False`                      |
| `annotator_id`          | `String`   | ID of user who edited results         | Optional (`nullable=True`)            |
| `annotation_date`       | `DateTime` | When annotation was made              | Default: `now()` via `server_default` |

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


