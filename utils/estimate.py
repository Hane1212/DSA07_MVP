import streamlit as st
import requests
import io
import pandas as pd
import os


# ----------------------------
# External API - Sree charan Lagudu
# ----------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000") 
# --- Helper to get auth headers ---
def get_auth_headers():
    if st.session_state.logged_in and st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}

def get_agmarknet_price(crop_name: str):
    api_key = st.secrets["AGMARKNET_API_KEY"]
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": api_key,
        "format": "json",
        "filters[crop]": crop_name,
        "limit": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if "records" in data and len(data["records"]) > 0:
        price = data["records"][0]["modal_price"]
        return float(price)
    else:
        raise Exception("Crop price not available.")
    
def get_weather(lat, lon):
    api_key = st.secrets["OPENWEATHER_API_KEY"]
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            temp = data['main']['temp'] - 273.15  # Kelvin → Celsius
            humidity = data['main']['humidity']
            return temp, humidity
        else:
            return None, None
    except:
        return None, None

def get_weather_data(city_name: str, api_key: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code != 200 or 'main' not in data:
        raise Exception("Weather API error")

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    weather = data['weather'][0]['description']
    return temp, humidity, weather


def get_crop_price(crop_name, state="Karnataka", district="Bangalore"):
    api_key =  st.secrets["AGMARKNET_API_KEY"]
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={api_key}&format=json&filters[commodity]={crop_name}&filters[state]={state}&filters[district]={district}&limit=1"

    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            if data["records"]:
                price = float(data["records"][0]["modal_price"])
                return price
    except Exception as e:
        print("Error fetching price:", e)
    return None

def show_estimate_for_file(file, result):
    detections = result.get("detections", [])
    model_name = result.get("model_name", "Unknown")

    if not detections:
        st.warning("No detections found for this file.")
        return

    # --- Display Detection Info ---
    st.subheader(f"Image: `{file.name}`")
    st.write(f"Model: **{model_name}** | Total detections: **{len(detections)}**")

    df = pd.DataFrame(detections)
    st.download_button("📄 Download CSV Report", df.to_csv(index=False).encode('utf-8'), f"{file.name}_report.csv", mime="text/csv")
    st.download_button("🌐 Download HTML Report", df.to_html(index=False), f"{file.name}_report.html", mime="text/html")

    st.markdown("---")
    st.subheader("📈 Estimated Yield Forecast")

    # --- Weather ---
    city = st.text_input("City for weather", "Bangalore", key=f"city_{file.name}")
    weather_api_key = st.secrets["OPENWEATHER_API_KEY"]
    temp, humidity, weather = get_weather_data(city, weather_api_key)
    if weather:
        st.info(f"🌡️ Temp: {temp:.2f}°C | 💧 Humidity: {humidity}%")
    else:
        st.warning("⚠️ Could not fetch real-time weather.")

    # --- Crop & Price ---
    crop_name = st.selectbox("Select Crop", ["Apple", "Mango", "Banana"], key=f"crop_{file.name}")
    market_price = get_crop_price(crop_name)

    # --- User Inputs ---
    avg_weight = st.number_input("Avg fruit weight (g)", 50, 500, 150)
    tree_count = st.number_input("Number of trees", 1, 10000, 100)
    fruit_count = len(detections)

    # --- Yield & Revenue Calculation ---
    estimated_yield_kg = (fruit_count * avg_weight * tree_count) / 1000
    st.success(f"✅ Estimated Yield: {estimated_yield_kg:.2f} kg")

    if market_price:
        revenue = estimated_yield_kg * market_price
        st.success(f"💰 Market Price: ₹{market_price:.2f} / kg")
        st.success(f"📊 Estimated Revenue: ₹{revenue:,.2f}")
    else:
        st.warning("❌ Market price not available.")

    # --- SMS Feature ---
    st.markdown("---")
    st.subheader("📱 Send SMS Alert")
    phone = st.text_input("Phone Number", key=f"sms_{file.name}")
    if st.button("📨 Send SMS Alert", key=f"sms_btn_{file.name}"):
        sms_data = {
            "username": st.session_state.get("username", "anonymous"),
            "file_name": file.name,
            "yield_kg": estimated_yield_kg,
            "phone_number": phone
        }
        try:
            res = requests.post(f"{FASTAPI_URL}/send_sms", json=sms_data, headers=get_auth_headers())
            res.raise_for_status()
            st.success("✅ SMS sent successfully!")
        except Exception as e:
            st.error(f"❌ SMS failed: {e}")
