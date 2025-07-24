# utils/estimate.py - Fixed version with proper API key handling

import streamlit as st
import requests
import io
import pandas as pd
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

# Add this at the top of utils/estimate.py
try:
    from .enhanced_quality import ExportQualityAssessment
    from .enhanced_pricing import PriceForecastingSystem
    ENHANCED_MODULES_AVAILABLE = True
except ImportError:
    ENHANCED_MODULES_AVAILABLE = False
    print("Enhanced modules not available, using basic functions only")

# ----------------------------
# FIXED API KEY HANDLING
# ----------------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000") 

def get_api_keys():
    """Centralized API key retrieval with multiple fallback methods"""
    api_keys = {}
    
    # Method 1: Try Streamlit secrets
    try:
        api_keys['OPENWEATHER_API_KEY'] = st.secrets["OPENWEATHER_API_KEY"]
        api_keys['AGMARKNET_API_KEY'] = st.secrets["AGMARKNET_API_KEY"]
        print("✅ API keys loaded from Streamlit secrets")
        return api_keys
    except Exception as e:
        print(f"⚠️ Streamlit secrets not available: {e}")
    
    # Method 2: Try environment variables
    try:
        weather_key = os.getenv("OPENWEATHER_API_KEY")
        agmark_key = os.getenv("AGMARKNET_API_KEY")
        
        if weather_key and agmark_key:
            api_keys['OPENWEATHER_API_KEY'] = weather_key
            api_keys['AGMARKNET_API_KEY'] = agmark_key
            print("✅ API keys loaded from environment variables")
            return api_keys
    except Exception as e:
        print(f"⚠️ Environment variables not available: {e}")
    
    # Method 3: Hardcoded fallback (for development only)
    api_keys = {
        'OPENWEATHER_API_KEY': "ae455fc340a8c66c34dc76e9121bcc66",
        'AGMARKNET_API_KEY': "579b464db66ec23bdd000001cbdf5ded3d654a61639a30e3ba03da1d"
    }
    print("⚠️ Using hardcoded API keys - not recommended for production")
    return api_keys

# Get API keys at module level
API_KEYS = get_api_keys()

# --- Helper to get auth headers ---
def get_auth_headers():
    if st.session_state.logged_in and st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}

def get_weather(lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
    """Get weather data by coordinates - FIXED VERSION"""
    try:
        api_key = API_KEYS.get('OPENWEATHER_API_KEY')
        if not api_key:
            print("❌ OpenWeather API key not available")
            return None, None
            
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        print(f"🌤️ Calling weather API: {url[:50]}...")
        
        res = requests.get(url, timeout=10)
        print(f"Weather API response status: {res.status_code}")
        
        if res.status_code == 200:
            data = res.json()
            temp = data['main']['temp'] - 273.15  # Kelvin → Celsius
            humidity = data['main']['humidity']
            print(f"✅ Weather data retrieved: {temp:.1f}°C, {humidity}%")
            return temp, humidity
        else:
            print(f"❌ Weather API error: {res.status_code} - {res.text}")
            return None, None
    except Exception as e:
        print(f"❌ Error in get_weather: {e}")
        return None, None

def get_weather_data(city_name: str, api_key: str = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Get weather data by city name - FIXED VERSION"""
    try:
        if not api_key:
            api_key = API_KEYS.get('OPENWEATHER_API_KEY')
            
        if not api_key:
            print("❌ OpenWeather API key not available")
            return None, None, None
            
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
        print(f"🌤️ Calling weather API for {city_name}: {url[:50]}...")
        
        response = requests.get(url, timeout=10)
        print(f"Weather API response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Weather API error: {response.status_code} - {response.text}")
            return None, None, None
            
        data = response.json()
        
        if 'main' not in data:
            print(f"❌ Invalid weather data structure: {data}")
            return None, None, None

        temp = data['main']['temp']
        humidity = data['main']['humidity']
        weather = data['weather'][0]['description']
        
        print(f"✅ Weather data for {city_name}: {temp}°C, {humidity}%, {weather}")
        return temp, humidity, weather
        
    except Exception as e:
        print(f"❌ Error in get_weather_data: {e}")
        return None, None, None

def get_crop_price(crop_name: str, state: str = "Karnataka", district: str = "Bangalore") -> Optional[float]:
    """Get crop price with FIXED API key handling"""
    try:
        api_key = API_KEYS.get('AGMARKNET_API_KEY')
        if not api_key:
            print("❌ AgMarkNet API key not available, using default prices")
            return get_default_price(crop_name)
            
        url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            'api-key': api_key,
            'format': 'json',
            'filters[commodity]': crop_name,
            'filters[state]': state,
            'filters[district]': district,
            'limit': 1
        }
        
        print(f"💰 Calling AgMarkNet API for {crop_name}...")
        res = requests.get(url, params=params, timeout=15)
        print(f"AgMarkNet API response status: {res.status_code}")

        if res.status_code == 200:
            data = res.json()
            if data.get("records") and len(data["records"]) > 0:
                price = float(data["records"][0]["modal_price"]) / 100  # Convert to Rs/kg
                print(f"✅ Live price for {crop_name}: ₹{price:.2f}/kg")
                return price
            else:
                print(f"⚠️ No price records found for {crop_name}")
        else:
            print(f"❌ AgMarkNet API error: {res.status_code} - {res.text}")
            
    except Exception as e:
        print(f"❌ Error fetching price for {crop_name}: {e}")
    
    # Fallback to default prices
    return get_default_price(crop_name)

def get_default_price(crop_name: str) -> float:
    """Get default prices with seasonal adjustment"""
    default_prices = {
        "Apple": 45.50,
        "Mango": 35.20,
        "Banana": 25.80,
        "Orange": 30.00,
        "Grapes": 55.00,
        "Pomegranate": 60.00
    }
    
    base_price = default_prices.get(crop_name, 40.00)
    
    # Apply seasonal factor
    seasonal_factors = {
        1: 1.2, 2: 1.15, 3: 1.1,    # Winter - higher prices
        4: 0.9, 5: 0.85, 6: 0.8,    # Spring - lower prices
        7: 0.75, 8: 0.8, 9: 0.85,   # Summer - harvest season
        10: 0.9, 11: 1.0, 12: 1.1   # Fall - pre-winter
    }
    
    seasonal_factor = seasonal_factors.get(datetime.now().month, 1.0)
    adjusted_price = base_price * seasonal_factor
    
    print(f"📊 Using default price for {crop_name}: ₹{adjusted_price:.2f}/kg (seasonal factor: {seasonal_factor})")
    return adjusted_price

def calculate_revenue(yield_kg: float, price_per_kg: float) -> float:
    """Calculate revenue based on yield and price"""
    return yield_kg * price_per_kg

def estimate_fruit_size(bboxes: List[Tuple[int, int, int, int]]) -> str:
    """Estimate fruit size based on bounding box areas"""
    if not bboxes:
        return "Unknown"
    
    sizes = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        area = abs((x2 - x1) * (y2 - y1))  # Use abs to handle negative values
        if area < 1500:
            sizes.append("Small")
        elif area < 3000:
            sizes.append("Medium")
        else:
            sizes.append("Large")
    
    # Return the most common size
    if sizes:
        return max(set(sizes), key=sizes.count)
    return "Unknown"

def estimate_ripeness(image_path: str) -> str:
    """Estimate ripeness based on color analysis"""
    try:
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return "Unknown"
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return "Unknown"
            
        # Convert BGR to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (ripe fruits)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Calculate ratio of red pixels
        total_pixels = img.shape[0] * img.shape[1]
        red_pixels = cv2.countNonZero(red_mask)
        ratio = red_pixels / total_pixels
        
        if ratio < 0.05:
            return "Unripe"
        elif ratio < 0.15:
            return "Semi-Ripe"
        else:
            return "Ripe"
            
    except Exception as e:
        print(f"Error estimating ripeness: {e}")
        return "Unknown"

def estimate_harvest_date(ripeness: str) -> str:
    """Estimate harvest date based on ripeness"""
    days_mapping = {
        "Ripe": 3,
        "Semi-Ripe": 7,
        "Unripe": 14,
        "Unknown": 7
    }
    
    days = days_mapping.get(ripeness, 7)
    harvest_date = datetime.now() + timedelta(days=days)
    return harvest_date.strftime('%Y-%m-%d')

def show_estimate_for_file(file, result):
    """Display estimation results for a file - ENHANCED VERSION"""
    detections = result.get("detections", [])
    model_name = result.get("model_name", "Unknown")

    if not detections:
        st.warning("No detections found for this file.")
        return

    # --- Display Detection Info ---
    st.subheader(f"Image: `{file.name}`")
    st.write(f"Model: **{model_name}** | Total detections: **{len(detections)}**")

    # Create DataFrame and add download buttons
    df = pd.DataFrame(detections)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Download CSV Report", 
            df.to_csv(index=False).encode('utf-8'), 
            f"{file.name}_report.csv", 
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "🌐 Download HTML Report", 
            df.to_html(index=False), 
            f"{file.name}_report.html", 
            mime="text/html"
        )

    st.markdown("---")
    st.subheader("📈 Estimated Yield Forecast")

    # --- Weather Section with FIXED API ---
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.text_input("City for weather", "Bangalore", key=f"city_{file.name}")
    
    with col2:
        if st.button("🌤️ Get Weather", key=f"weather_btn_{file.name}"):
            with st.spinner("Fetching weather data..."):
                temp, humidity, weather = get_weather_data(city)
                if temp is not None:
                    st.success(f"🌡️ Temp: {temp:.1f}°C | 💧 Humidity: {humidity}% | {weather}")
                    # Store weather data for use in enhanced analysis
                    st.session_state[f"weather_{file.name}"] = {
                        'temp': temp, 'humidity': humidity, 'weather': weather
                    }
                else:
                    st.error("⚠️ Could not fetch weather data - check API key configuration")

    # --- Crop Selection and Price with FIXED API ---
    col1, col2 = st.columns(2)
    
    with col1:
        crop_name = st.selectbox(
            "Select Crop", 
            ["Apple", "Mango", "Banana", "Orange", "Grapes"], 
            key=f"crop_{file.name}"
        )
    
    with col2:
        if st.button("💰 Get Live Price", key=f"price_btn_{file.name}"):
            with st.spinner("Fetching live price..."):
                market_price = get_crop_price(crop_name)
                if market_price:
                    st.success(f"💰 Market Price: ₹{market_price:.2f}/kg")
                    st.session_state[f"price_{file.name}"] = market_price
                else:
                    st.error("⚠️ Could not fetch live price - using default price")
                    st.session_state[f"price_{file.name}"] = get_default_price(crop_name)

    # Get stored price or default
    market_price = st.session_state.get(f"price_{file.name}", get_crop_price(crop_name))

    # --- User Inputs ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_weight = st.number_input(
            "Avg fruit weight (g)", 
            min_value=50, 
            max_value=500, 
            value=150, 
            key=f"avg_weight_{file.name}"
        )
    
    with col2:
        tree_count = st.number_input(
            "Number of trees", 
            min_value=1, 
            max_value=10000, 
            value=100, 
            key=f"tree_count_{file.name}"
        )
    
    with col3:
        fruit_count = len(detections) if detections else st.number_input(
            "Fruit count (manual)", 
            min_value=0,
            value=20, 
            key=f"fruit_count_{file.name}"
        )

    # --- Calculations ---
    estimated_yield_kg = (fruit_count * avg_weight * tree_count) / 1000
    
    # Display results
    st.markdown("### 📊 Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🍎 Total Fruits", f"{fruit_count:,}")
    
    with col2:
        st.metric("📦 Estimated Yield", f"{estimated_yield_kg:.2f} kg")
    
    with col3:
        if market_price:
            revenue = calculate_revenue(estimated_yield_kg, market_price)
            st.metric("💰 Estimated Revenue", f"₹{revenue:,.2f}")

    # --- ENHANCED ANALYSIS SECTION ---
    if ENHANCED_MODULES_AVAILABLE:
        st.markdown("---")
        st.subheader("🏆 Enhanced Analysis")
        
        if st.button("🚀 Run Enhanced Analysis", key=f"enhanced_{file.name}"):
            with st.spinner("Performing enhanced analysis..."):
                try:
                    quality_assessor = ExportQualityAssessment()
                    price_forecaster = PriceForecastingSystem()
                    
                    # Create temporary image path if needed
                    temp_image_path = f"./temp/images/{file.name}"
                    
                    # Enhanced quality assessment
                    quality = quality_assessor.assess_export_quality(detections, temp_image_path)
                    
                    # Enhanced price forecasting
                    forecasting = price_forecaster.forecast_prices(crop_name, estimated_yield_kg, quality.grade)
                    
                    # Display enhanced results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🏆 Quality Grade", quality.grade.split()[0])
                    
                    with col2:
                        st.metric("📦 Export Ready", f"{quality.export_ready_percentage:.1f}%")
                    
                    with col3:
                        st.metric("💰 Enhanced Price", f"₹{forecasting.current_price:.2f}/kg")
                    
                    with col4:
                        st.metric("📈 Revenue Forecast", f"₹{forecasting.revenue_estimate:,.2f}")
                    
                    # Additional enhanced metrics
                    st.markdown("#### 🔍 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **Quality Metrics:**
                        - Ripeness Score: {quality.ripeness_score:.2f}
                        - Defect Rate: {quality.defect_rate:.1%}
                        - Shelf Life: {quality.shelf_life_days} days
                        - Harvest Date: {quality.harvest_date}
                        """)
                    
                    with col2:
                        st.info(f"""
                        **Market Forecast:**
                        - 7-Day Price: ₹{forecasting.predicted_price_7d:.2f}/kg
                        - 30-Day Price: ₹{forecasting.predicted_price_30d:.2f}/kg
                        - Market Trend: {forecasting.market_trend}
                        - Confidence: {forecasting.confidence_score:.1%}
                        """)
                    
                    st.success("✅ Enhanced analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Enhanced analysis failed: {str(e)}")
                    st.error("Please check your API keys and module installation.")
    else:
        st.warning("⚠️ Enhanced analysis modules not available. Install enhanced modules to access advanced features.")

    # --- Additional Estimations ---
    if result.get("detections"):
        st.markdown("### 🔍 Fruit Analysis")
        
        # Extract bounding boxes for analysis
        bboxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3]) for d in detections]
        size = estimate_fruit_size(bboxes)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📏 **Size**: {size}")
        
        with col2:
            # For ripeness, we'd need the actual image path
            ripeness = result.get("ripeness", "Unknown")
            st.info(f"🍃 **Ripeness**: {ripeness}")
        
        with col3:
            harvest_date = estimate_harvest_date(ripeness)
            st.info(f"📅 **Harvest Date**: {harvest_date}")

    # --- SMS Feature ---
    st.markdown("---")
    st.subheader("📱 Send SMS Alert")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        phone = st.text_input("Phone Number", key=f"sms_{file.name}")
    
    with col2:
        st.write("")  # Empty space for alignment
        if st.button("📨 Send SMS Alert", key=f"sms_btn_{file.name}"):
            if not phone:
                st.error("Please enter a phone number")
            else:
                sms_data = {
                    "username": st.session_state.get("username", "anonymous"),
                    "file_name": file.name,
                    "yield_kg": estimated_yield_kg,
                    "phone_number": phone,
                    "crop_name": crop_name,
                    "revenue": revenue if market_price else 0
                }
                try:
                    res = requests.post(
                        f"{FASTAPI_URL}/send_sms", 
                        json=sms_data, 
                        headers=get_auth_headers(),
                        timeout=10
                    )
                    res.raise_for_status()
                    st.success("✅ SMS sent successfully!")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ SMS failed: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

# Test function to verify API keys
def test_api_connections():
    """Test function to verify API key connections"""
    st.subheader("🔧 API Connection Test")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Weather API"):
            temp, humidity, weather = get_weather_data("Bangalore")
            if temp is not None:
                st.success(f"✅ Weather API working: {temp}°C, {humidity}%, {weather}")
            else:
                st.error("❌ Weather API failed")
    
    with col2:
        if st.button("Test Price API"):
            price = get_crop_price("Apple")
            if price:
                st.success(f"✅ Price API working: ₹{price:.2f}/kg")
            else:
                st.error("❌ Price API failed")
    
    # Display current API key status
    st.markdown("#### API Key Status:")
    weather_key = API_KEYS.get('OPENWEATHER_API_KEY', 'Not set')
    agmark_key = API_KEYS.get('AGMARKNET_API_KEY', 'Not set')
    
    st.code(f"""
Weather API Key: {weather_key[:10]}...{weather_key[-5:] if len(weather_key) > 15 else weather_key}
AgMarkNet API Key: {agmark_key[:10]}...{agmark_key[-5:] if len(agmark_key) > 15 else agmark_key}
    """)