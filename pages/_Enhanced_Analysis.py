import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import sys
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

sys.path.append('..')

# Import your enhanced modules (with error handling)
try:
    from utils.enhanced_quality import ExportQualityAssessment
    from utils.enhanced_pricing import PriceForecastingSystem  
    from utils.enhanced_dashboard import create_quality_dashboard, create_price_dashboard
except ImportError as e:
    st.error(f"Enhanced modules not available: {e}")
    # Create mock classes for development
    class ExportQualityAssessment:
        def assess_export_quality(self, detections, image_path):
            return type('obj', (object,), {
                'grade': 'A',
                'export_ready_percentage': 85.0,
                'defect_rate': 5.0,
                'size_distribution': {'Small': 20, 'Medium': 60, 'Large': 20},
                'ripeness_score': 0.8,
                'shelf_life_days': 14
            })()
    
    class PriceForecastingSystem:
        def forecast_prices(self, crop, yield_kg, grade):
            return type('obj', (object,), {
                'current_price': 45.50,
                'predicted_price_7d': 47.50,
                'predicted_price_30d': 50.00,
                'market_trend': 'increasing',
                'supply_demand_ratio': 0.95,
                'price_volatility': 0.15,
                'confidence_score': 0.85
            })()

def create_quality_dashboard(quality_data):
    """Create quality assessment dashboard"""
    st.subheader("🍎 Export Quality Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Quality Grade", quality_data.grade, delta="Premium" if quality_data.grade == 'A' else "Standard")
    
    with col2:
        st.metric("Export Ready", f"{quality_data.export_ready_percentage:.1f}%")
    
    with col3:
        st.metric("Defect Rate", f"{quality_data.defect_rate:.1f}%", delta=f"-{quality_data.defect_rate:.1f}%")
    
    with col4:
        st.metric("Shelf Life", f"{quality_data.shelf_life_days} days")
    
    # Size distribution chart
    if hasattr(quality_data, 'size_distribution'):
        fig = px.pie(
            values=list(quality_data.size_distribution.values()),
            names=list(quality_data.size_distribution.keys()),
            title="Size Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Ripeness score gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = quality_data.ripeness_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Ripeness Score"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    st.plotly_chart(fig_gauge, use_container_width=True)

def create_price_dashboard(price_data, yield_kg):
    """Create price forecasting dashboard"""
    st.subheader("💰 Price Forecasting & Revenue Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${price_data.current_price:.2f}/kg")
    
    with col2:
        st.metric("7-Day Forecast", f"${price_data.predicted_price_7d:.2f}/kg", 
                 delta=f"{((price_data.predicted_price_7d/price_data.current_price-1)*100):+.1f}%")
    
    with col3:
        st.metric("30-Day Forecast", f"${price_data.predicted_price_30d:.2f}/kg",
                 delta=f"{((price_data.predicted_price_30d/price_data.current_price-1)*100):+.1f}%")
    
    with col4:
        revenue = yield_kg * price_data.current_price
        st.metric("Estimated Revenue", f"${revenue:.2f}")
    
    # Price trend chart
    dates = [datetime.now() + timedelta(days=i) for i in range(31)]
    current_price = price_data.current_price
    price_7d = price_data.predicted_price_7d
    price_30d = price_data.predicted_price_30d
    
    # Create price trend line
    prices = []
    for i, date in enumerate(dates):
        if i == 0:
            prices.append(current_price)
        elif i <= 7:
            # Linear interpolation to 7-day price
            prices.append(current_price + (price_7d - current_price) * (i/7))
        else:
            # Linear interpolation from 7-day to 30-day price
            prices.append(price_7d + (price_30d - price_7d) * ((i-7)/(30-7)))
    
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Price Forecast',
        line=dict(color='blue', width=3)
    ))
    
    fig_price.update_layout(
        title="Price Forecast (30 Days)",
        xaxis_title="Date",
        yaxis_title="Price ($/kg)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Market indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Trend", price_data.market_trend.title(), 
                 delta="Bullish" if price_data.market_trend == "increasing" else "Bearish")
    
    with col2:
        st.metric("Supply/Demand Ratio", f"{price_data.supply_demand_ratio:.2f}")
    
    with col3:
        st.metric("Price Volatility", f"{price_data.price_volatility:.1%}")

def main_analysis_interface():
    st.title("📊 Enhanced Fruit Analysis")
    st.markdown("Advanced quality assessment and price forecasting for agricultural produce")
    
    # Sample data simulation (replace with real detection results)
    st.sidebar.header("Analysis Settings")
    
    # Mock detection data for demonstration
    sample_detections = [
        {"label": "apple", "score": 0.95, "box": [100, 100, 200, 200]},
        {"label": "apple", "score": 0.87, "box": [250, 150, 350, 250]},
        {"label": "apple", "score": 0.92, "box": [400, 120, 500, 220]}
    ]
    
    fruit_count = len(sample_detections)
    estimated_yield = fruit_count * 0.15  # 150g per apple
    
    if st.sidebar.button("Run Enhanced Analysis"):
        with st.spinner("Performing enhanced analysis..."):
            # Initialize systems
            quality_assessor = ExportQualityAssessment()
            price_forecaster = PriceForecastingSystem()
            
            # Run assessments
            quality_results = quality_assessor.assess_export_quality(sample_detections, "temp_image.jpg")
            price_results = price_forecaster.forecast_prices("Apple", estimated_yield, quality_results.grade)
            
            # Create dashboards
            create_quality_dashboard(quality_results)
            st.divider()
            create_price_dashboard(price_results, estimated_yield)
            
            # Additional insights
            st.subheader("📈 Market Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Quality Summary:**
                - Grade: {quality_results.grade}
                - Export Ready: {quality_results.export_ready_percentage:.1f}%
                - Estimated Shelf Life: {quality_results.shelf_life_days} days
                """)
            
            with col2:
                trend_emoji = "📈" if price_results.market_trend == "increasing" else "📉"
                st.info(f"""
                **Price Summary:**
                - Current: ${price_results.current_price:.2f}/kg
                - Trend: {trend_emoji} {price_results.market_trend.title()}
                - Confidence: {price_results.confidence_score:.1%}
                """)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if quality_results.grade == 'A' and price_results.market_trend == "increasing":
                st.success("🎯 **Optimal conditions**: High quality produce with rising prices. Consider immediate harvest and export.")
            elif quality_results.export_ready_percentage < 70:
                st.warning("⚠️ **Quality concerns**: Consider additional sorting or processing before export.")
            elif price_results.predicted_price_30d > price_results.current_price * 1.1:
                st.info("💰 **Price opportunity**: Prices expected to rise significantly. Consider delayed harvest if quality permits.")
            else:
                st.info("📊 **Standard conditions**: Proceed with normal harvest and export planning.")
    
    else:
        # Show placeholder content
        st.info("Click 'Run Enhanced Analysis' in the sidebar to generate detailed quality and price analysis.")
        
        # Show sample charts with placeholder data
        st.subheader("Sample Analysis Preview")
        
        # Sample quality metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality Grade", "A", delta="Premium")
        with col2:
            st.metric("Export Ready", "85.0%")
        with col3:
            st.metric("Defect Rate", "5.0%", delta="-5.0%")
        with col4:
            st.metric("Shelf Life", "14 days")
        
        # Sample price chart
        dates = [datetime.now() + timedelta(days=i) for i in range(8)]
        prices = [45.50, 46.20, 46.80, 47.50, 47.20, 47.80, 48.00, 48.50]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name='Price Forecast'))
        fig.update_layout(title="Sample Price Forecast", xaxis_title="Date", yaxis_title="Price ($/kg)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main_analysis_interface()