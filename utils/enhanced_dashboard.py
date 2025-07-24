try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, timedelta
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