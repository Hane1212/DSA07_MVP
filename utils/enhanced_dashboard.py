try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
    from utils.enhanced_quality import ExportQualityAssessment
    from utils.enhanced_pricing import PriceForecastingSystem
    from utils.enhanced_dashboard import create_quality_dashboard, create_price_dashboard
except ImportError as e:
    print(f"Dashboard dependencies not available: {e}")


def create_quality_dashboard(quality):
    """Create export quality dashboard"""
    
    st.subheader("🏆 Export Quality Assessment")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🍎 Fruit Count", quality.fruit_count)
    
    with col2:
        st.metric("📏 Average Size", quality.avg_size)
    
    with col3:
        st.metric("🍃 Ripeness", quality.ripeness)
    
    with col4:
        st.metric("⭐ Export Grade", quality.grade.split()[0])
    
    # Quality details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Size Distribution")
        if quality.size_distribution:
            fig = px.pie(
                values=list(quality.size_distribution.values()),
                names=list(quality.size_distribution.keys()),
                title="Fruit Size Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Metrics")
        
        # Quality progress bars
        st.metric("🟢 Ripeness Score", f"{quality.ripeness_score:.2f}")
        st.progress(quality.ripeness_score)
        
        st.metric("🔴 Defect Rate", f"{quality.defect_rate:.1%}")
        st.progress(1 - quality.defect_rate)
        
        st.metric("📦 Export Ready", f"{quality.export_ready_percentage:.1f}%")
        st.progress(quality.export_ready_percentage / 100)
    
    # Harvest information
    st.subheader("📅 Harvest Planning")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Optimal Harvest Date**\n{quality.harvest_date}")
    
    with col2:
        st.info(f"**Shelf Life**\n{quality.shelf_life_days} days")
    
    with col3:
        try:
            days_until_harvest = (datetime.strptime(quality.harvest_date, '%Y-%m-%d') - datetime.now()).days
            st.info(f"**Days Until Harvest**\n{max(0, days_until_harvest)} days")
        except:
            st.info("**Days Until Harvest**\nCalculating...")

def create_price_dashboard(forecasting, crop_name: str):
    """Create price forecasting dashboard"""
    
    st.subheader("💰 Price Forecasting & Market Analysis")
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💵 Current Price", f"₹{forecasting.current_price:.2f}/kg")
    
    with col2:
        price_change_7d = forecasting.predicted_price_7d - forecasting.current_price
        st.metric(
            "📈 7-Day Forecast", 
            f"₹{forecasting.predicted_price_7d:.2f}/kg",
            delta=f"₹{price_change_7d:.2f}"
        )
    
    with col3:
        price_change_30d = forecasting.predicted_price_30d - forecasting.current_price
        st.metric(
            "📊 30-Day Forecast", 
            f"₹{forecasting.predicted_price_30d:.2f}/kg",
            delta=f"₹{price_change_30d:.2f}"
        )
    
    with col4:
        st.metric("💰 Revenue Estimate", f"₹{forecasting.revenue_estimate:,.2f}")
    
    # Market analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Price Trend Analysis")
        
        # Create price trend chart
        dates = [
            datetime.now().strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        ]
        prices = [
            forecasting.current_price,
            forecasting.predicted_price_7d,
            forecasting.predicted_price_30d
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines+markers',
            name='Price Forecast',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title=f"{crop_name} Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (₹/kg)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Market Indicators")
        
        # Market trend indicator
        trend_color = "green" if forecasting.market_trend == "Upward" else "red" if forecasting.market_trend == "Downward" else "blue"
        st.markdown(f"**Trend:** <span style='color:{trend_color}'>{forecasting.market_trend}</span>", unsafe_allow_html=True)
        
        # Other indicators
        st.metric("🌾 Supply/Demand Ratio", f"{forecasting.supply_demand_ratio:.2f}")
        st.metric("📊 Price Volatility", f"{forecasting.price_volatility:.1%}")
        st.metric("🌸 Seasonal Factor", f"{forecasting.seasonal_factor:.2f}")
        st.metric("🎯 Confidence Score", f"{forecasting.confidence_score:.1%}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if forecasting.market_trend == "Upward":
            st.success("🟢 Consider delaying harvest for better prices")
        elif forecasting.market_trend == "Downward":
            st.warning("🟡 Consider early harvest or storage")
        else:
            st.info("🔵 Current timing seems optimal")

def enhanced_results_display(ENHANCED_MODULES_AVAILABLE):
    if ENHANCED_MODULES_AVAILABLE and 'enhanced_quality_result' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Latest Analysis")
        quality = st.session_state.enhanced_quality_result
        price = st.session_state.enhanced_price_result
        
        st.sidebar.metric("Quality Grade", quality.grade.split()[0])
        st.sidebar.metric("Export Ready", f"{quality.export_ready_percentage:.1f}%")
        st.sidebar.metric("Current Price", f"₹{price.current_price:.2f}/kg")
        st.sidebar.metric("Revenue Est.", f"₹{price.revenue_estimate:,.0f}")

def enhanced_analysis(ENHANCED_MODULES_AVAILABLE, tab_enhanced):
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
    