import streamlit as st
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from utils.chatbox import show_chatbox
from utils.style import custom_css
from utils import estimate
from utils import utils

# Import enhanced modules with error handling
try:
    from utils.enhanced_quality import ExportQualityAssessment
    from utils.enhanced_pricing import PriceForecastingSystem
    from utils.enhanced_dashboard import create_quality_dashboard, create_price_dashboard
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fruit Counter MVP", layout="wide")

# --- PAGE SWITCH FUNCTION (REQUIRES streamlit>=1.12) ---
def go_to(page_name: str):
    st.switch_page(f"pages/{page_name}")

# --- CUSTOM CSS STYLE ---
st.markdown(custom_css, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = "" # "admin", "annotator", "viewer"
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = ""

# --- Navigation ---
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/_Compare.py", label="📊 Compare Models")
st.sidebar.page_link("pages/_Login.py", label="🔑 User Authentication")

# --- Tabs ---
if ENHANCED_MODULES_AVAILABLE:
    tab_detection, tab_chat, tab_his, tab_est, tab_enhanced = st.tabs([
        "🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs", "📈 Estimate", "🏆 Enhanced Analysis"
    ])
else:
    tab_detection, tab_chat, tab_his, tab_est = st.tabs([
        "🧠 Detection", "💬 AgriBot Chat", "🧾 History / Logs", "📈 Estimate"
    ])

# --- Initialize default values for export quality and price forecasting ---
# These should be calculated based on actual detection results
size = "Medium"  # Default value
ripeness = "Semi-Ripe"  # Default value
harvest_date = "2025-08-01"  # Default value
price = 45.50  # Default price in ₹ per kg
revenue = 2275.00  # Default revenue

# Check if we have actual detection results to update these values
if 'latest_detection_result' in st.session_state:
    result = st.session_state.latest_detection_result
    detections = result.get("detections", [])
    if detections:
        # Extract bounding boxes for size estimation
        bboxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3]) for d in detections]
        size = estimate.estimate_fruit_size(bboxes)
        
        # Get crop price and calculate revenue
        fruit_count = len(detections)
        avg_weight_grams = 150  # Default weight per fruit
        estimated_yield_kg = (fruit_count * avg_weight_grams) / 1000
        price = estimate.get_crop_price("Apple") or 45.50
        revenue = estimated_yield_kg * price

# Display basic metrics
st.subheader("📦 Export Quality")
st.write(f"Fruit Size: **{size}**")
st.write(f"Ripeness: **{ripeness}**")
st.write(f"Estimated Harvest Date: **{harvest_date}**")

st.subheader("💸 Price Forecasting")
st.write(f"Live Price: ₹{price:.2f} per kg")
st.write(f"Predicted Revenue: ₹{revenue:,.2f}")

# --- TAB 1: Detection ---
with tab_detection:
    st.header("Fruit Detection")
    utils.show_fruit_detection()

# --- TAB 2: Chatbot ---
with tab_chat:
    st.header("👩‍🌾 Ask AgriBot")
    show_chatbox()

# --- TAB 3: Historical Data ---
with tab_his:
    st.subheader("📜 Detection History")
    utils.show_detection_history()

# --- TAB 4: Estimate ---
with tab_est:
    st.subheader("🧮 Estimate")
    for key in st.session_state:
        if key.startswith("result_"):
            file_name = key.replace("result_", "")
            result = st.session_state[key]
            uploaded_file = st.session_state.get(f"file_{file_name}")
            image = st.session_state.get(f"image_{file_name}")
            
            if uploaded_file and image:
                estimate.show_estimate_for_file(uploaded_file, result)

# --- TAB 5: Enhanced Analysis ---
if ENHANCED_MODULES_AVAILABLE:
    with tab_enhanced:
        st.header("🏆 Enhanced Export Quality & Price Forecasting")
        
        # Initialize systems
        if 'quality_assessor' not in st.session_state:
            st.session_state.quality_assessor = ExportQualityAssessment()
        if 'price_forecaster' not in st.session_state:
            st.session_state.price_forecaster = PriceForecastingSystem()
        
        # Use existing detection results if available
        if 'latest_detection_result' in st.session_state:
            result = st.session_state.latest_detection_result
            detections = result.get("detections", [])
            
            if detections:
                st.success(f"✅ Found {len(detections)} detections from previous analysis")
                
                # Get image path
                image_path = result.get("image_path", "")
                
                # Crop selection
                crop_name = st.selectbox("Select Crop Type", 
                    ["Apple", "Mango", "Orange", "Banana", "Grapes"], 
                    key="enhanced_crop_select")
                
                # Analysis settings
                col1, col2 = st.columns(2)
                with col1:
                    avg_weight_grams = st.number_input("Average fruit weight (grams)", 
                                                     min_value=50, max_value=500, value=150)
                with col2:
                    quality_threshold = st.slider("Quality threshold", 0.0, 1.0, 0.7, 0.1)
                
                if st.button("🚀 Run Enhanced Analysis", type="primary"):
                    with st.spinner("Performing enhanced analysis..."):
                        try:
                            # Quality assessment
                            quality = st.session_state.quality_assessor.assess_export_quality(
                                detections, image_path
                            )
                            
                            # Calculate yield
                            estimated_yield_kg = (quality.fruit_count * avg_weight_grams) / 1000
                            
                            # Price forecasting  
                            forecasting = st.session_state.price_forecaster.forecast_prices(
                                crop_name, estimated_yield_kg, quality.grade
                            )
                            
                            # Display dashboards
                            st.markdown("---")
                            create_quality_dashboard(quality)
                            st.markdown("---")
                            create_price_dashboard(forecasting, crop_name)
                            
                            # Store results for future reference
                            st.session_state.enhanced_quality_result = quality
                            st.session_state.enhanced_price_result = forecasting
                            
                            # Action recommendations
                            st.markdown("---")
                            st.subheader("💡 Action Recommendations")
                            
                            # Generate recommendations based on results
                            recommendations = []
                            
                            if quality.export_ready_percentage >= 85:
                                recommendations.append("🌟 **Premium Quality**: Ready for premium export markets")
                            elif quality.export_ready_percentage >= 70:
                                recommendations.append("✅ **Good Quality**: Suitable for standard export")
                            else:
                                recommendations.append("⚠️ **Quality Concerns**: Consider additional processing")
                            
                            if forecasting.market_trend == "Upward":
                                recommendations.append("📈 **Price Trending Up**: Consider delaying harvest if possible")
                            elif forecasting.market_trend == "Downward":
                                recommendations.append("📉 **Price Declining**: Harvest and sell quickly")
                            else:
                                recommendations.append("📊 **Stable Prices**: Normal harvest timing recommended")
                            
                            if quality.shelf_life_days < 7:
                                recommendations.append("⏰ **Short Shelf Life**: Prioritize immediate processing")
                            
                            for rec in recommendations:
                                st.info(rec)
                                
                        except Exception as e:
                            st.error(f"Error during enhanced analysis: {str(e)}")
                            st.error("Please check your detection results and try again.")
            else:
                st.info("👆 Please run fruit detection first to enable enhanced analysis")
                st.markdown("### How to use Enhanced Analysis:")
                st.markdown("""
                1. Go to the **Detection** tab
                2. Upload an image and run fruit detection
                3. Return to this tab for detailed quality and price analysis
                """)
        else:
            st.info("👆 Please run fruit detection first to enable enhanced analysis")
            
            # Show sample analysis for demonstration
            if st.button("🎯 Show Demo Analysis"):
                st.markdown("---")
                st.subheader("📊 Demo Analysis Results")
                
                # Create sample data
                sample_detections = [
                    {"label": "apple", "score": 0.95, "box": [100, 100, 200, 200]},
                    {"label": "apple", "score": 0.87, "box": [250, 150, 350, 250]},
                    {"label": "apple", "score": 0.92, "box": [400, 120, 500, 220]}
                ]
                
                try:
                    # Demo quality assessment
                    quality = st.session_state.quality_assessor.assess_export_quality(
                        sample_detections, ""
                    )
                    
                    # Demo price forecasting
                    forecasting = st.session_state.price_forecaster.forecast_prices(
                        "Apple", 1.5, quality.grade
                    )
                    
                    create_quality_dashboard(quality)
                    st.markdown("---")
                    create_price_dashboard(forecasting, "Apple")
                    
                except Exception as e:
                    st.error(f"Demo analysis error: {str(e)}")

# --- Enhanced Results Display ---
if ENHANCED_MODULES_AVAILABLE and 'enhanced_quality_result' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Latest Analysis")
    quality = st.session_state.enhanced_quality_result
    price = st.session_state.enhanced_price_result
    
    st.sidebar.metric("Quality Grade", quality.grade.split()[0])
    st.sidebar.metric("Export Ready", f"{quality.export_ready_percentage:.1f}%")
    st.sidebar.metric("Current Price", f"₹{price.current_price:.2f}/kg")
    st.sidebar.metric("Revenue Est.", f"₹{price.revenue_estimate:,.0f}")

# --- CLOSE DIV ---
st.markdown("</div>", unsafe_allow_html=True)