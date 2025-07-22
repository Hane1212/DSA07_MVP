import requests
import streamlit as st
import pandas as pd

# Add mock data
MARKET_PRICES = {
    "apple": [0.4, 0.45, 0.5],
    "banana": [0.2, 0.25, 0.3],
    "orange": [0.35, 0.38, 0.4]
}

WORKLOAD_TIME = {
    "apple": 0.05,   # hours per fruit
    "banana": 0.03,
    "orange": 0.04
}


LOCAL_HOST = "http://fastapi:8000"

def run_detection_api(file, model_name):
    """
    Sends an image and model name to the /predict/ endpoint,
    adds fruit type and computed fields.
    """
    files = {"file": (file.name, file.getvalue(), "image/png")}
    data = {"model_name": model_name}
    response = requests.post(f"{LOCAL_HOST}/predict/", files=files, data=data)
    response.raise_for_status()
    
    result = response.json()  # ✅ capture the result first

    # Mock fruit type (replace with real detection later)
    fruit_type = result.get("fruit_type", "banana")

    # Inject computed fields
    result["fruit_type"] = fruit_type
    result["estimated_workload"] = estimate_workload(result["fruit_count"], fruit_type)
    result["market_price"] = get_latest_price(fruit_type)
    result["expected_revenue"], _ = calculate_revenue(result["fruit_count"], fruit_type)

    return result  # ✅ return AFTER adding new fields

def save_detection_api(result, model_name):
    """
    Sends detection metadata to the /save endpoint.
    """
    print("📤 Sending save request...")
    payload = {
        "image_id": result["image_id"],
        "image_path": result["image_path"],
        "model_name": model_name,
        "detection_time": result["detection_time"],
        "confidence": result["confidence"],
        "model_predictions": result.get("model_predictions", []),
        "user_annotations": [],
        "num_objects_model": result["fruit_count"],
        "num_objects_corrected": 0,
        "annotated_by_user": False,
        "annotator_id": None
    }

    response = requests.post(f"{LOCAL_HOST}/save", json=payload)
    print("📥 Response:", response.status_code, response.text)
    response.raise_for_status()
    return response.json()


def show_detection_history():
    with st.form("filter_form"):
        image_filter = st.text_input("🔍 Filter by image name (partial match)")
        min_count = st.number_input("🔢 Minimum object count", min_value=0, step=1)
        show_all = st.checkbox("📋 Show all (ignore filters)", value=False)
        submitted = st.form_submit_button("🖼️ Get Image")

    if submitted:
        try:
            params = {}
            if not show_all:
                if image_filter:
                    params["image_id"] = image_filter
                if min_count:
                    params["min_count"] = min_count

            response = requests.get(f"{LOCAL_HOST}/detections", params=params)
            data = response.json()

            if not data:
                st.info("No matching records found.")
                return

            # Build table
            rows = []
            for d in data:
                rows.append([
                    d.get("image_id"),
                    d.get("model_name"),
                    d.get("num_objects_model"),
                    d.get("num_objects_corrected"),
                    d.get("detection_time"),
                    f"{d.get('confidence') * 100:.2f}%"
                ])

            st.markdown("### 🧾 Results")
            st.table(
                pd.DataFrame(rows, columns=[
                    "🖼 Image Name", "🤖 Model", "🔢 Model Count", "✏️ Corrected Count",
                    "⏱ Detection Time (s)", "📏 Confidence"
                ])
            )

        except Exception as e:
            st.error("⚠️ Failed to load detection data.")
            st.exception(e)
def estimate_workload(fruit_count, fruit_type):
    time_per_fruit = WORKLOAD_TIME.get(fruit_type.lower(), 0.04)
    return round(fruit_count * time_per_fruit, 2)

def get_latest_price(fruit_type):
    return MARKET_PRICES.get(fruit_type.lower(), [0.2])[-1]

def calculate_revenue(fruit_count, fruit_type):
    price = get_latest_price(fruit_type)
    return round(price * fruit_count, 2), price
