import requests
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class PriceForecasting:
    """Data class for price forecasting metrics"""
    current_price: float
    predicted_price_7d: float
    predicted_price_30d: float
    market_trend: str
    seasonal_factor: float
    supply_demand_ratio: float
    price_volatility: float
    revenue_estimate: float
    confidence_score: float

class PriceForecastingSystem:
    """Enhanced price forecasting system"""
    
    def __init__(self):
        self.historical_prices = {}
        self.seasonal_factors = {
            1: 1.2, 2: 1.15, 3: 1.1,    # Winter - higher prices
            4: 0.9, 5: 0.85, 6: 0.8,    # Spring - lower prices
            7: 0.75, 8: 0.8, 9: 0.85,   # Summer - harvest season
            10: 0.9, 11: 1.0, 12: 1.1   # Fall - pre-winter
        }
    
    def get_current_market_price(self, crop_name: str, state: str = "Karnataka") -> float:
        """Get current market price with enhanced error handling"""
        try:
            # Try multiple API sources
            price = self._get_agmarknet_price(crop_name, state)
            if price:
                return price
            
            # Fallback to default prices with seasonal adjustment
            default_prices = {
                "Apple": 45.50, "Mango": 35.20, "Banana": 25.80,
                "Orange": 30.00, "Grapes": 55.00, "Pomegranate": 60.00
            }
            
            base_price = default_prices.get(crop_name, 40.00)
            seasonal_factor = self.seasonal_factors.get(datetime.now().month, 1.0)
            
            return base_price * seasonal_factor
            
        except Exception as e:
            print(f"Error getting market price: {e}")
            return 40.00
    
    def _get_agmarknet_price(self, crop_name: str, state: str) -> Optional[float]:
        """Internal method to fetch from AgMarkNet API"""
        try:
            # This would connect to actual AgMarkNet API
            # For demo purposes, return simulated data
            base_prices = {"Apple": 45.50, "Mango": 35.20, "Banana": 25.80}
            return base_prices.get(crop_name, 40.00)
        except:
            return None
    
    def predict_price_trends(self, crop_name: str, current_price: float, yield_kg: float) -> Tuple[float, float, str]:
        """Predict price trends for 7 and 30 days"""
        
        # Market factors simulation
        supply_factor = min(1.2, 1000 / max(yield_kg, 100))  # Higher yield = lower price
        demand_factor = np.random.uniform(0.95, 1.05)  # Random demand fluctuation
        seasonal_factor = self.seasonal_factors.get(datetime.now().month, 1.0)
        
        # 7-day prediction
        price_7d = current_price * supply_factor * demand_factor * 0.98
        
        # 30-day prediction with more uncertainty
        trend_factor = np.random.uniform(0.9, 1.1)
        price_30d = current_price * supply_factor * seasonal_factor * trend_factor
        
        # Determine trend
        if price_30d > current_price * 1.05:
            trend = "Upward"
        elif price_30d < current_price * 0.95:
            trend = "Downward"
        else:
            trend = "Stable"
        
        return price_7d, price_30d, trend
    
    def calculate_market_metrics(self, yield_kg: float, region: str = "South India") -> Tuple[float, float, float]:
        """Calculate advanced market metrics"""
        
        # Supply-demand ratio (simulated)
        regional_supply = np.random.uniform(800, 1200)  # tonnes
        regional_demand = np.random.uniform(900, 1100)  # tonnes
        supply_demand_ratio = regional_supply / regional_demand
        
        # Price volatility (last 30 days simulation)
        volatility = np.random.uniform(0.05, 0.15)
        
        # Seasonal factor
        seasonal_factor = self.seasonal_factors.get(datetime.now().month, 1.0)
        
        return supply_demand_ratio, volatility, seasonal_factor
    
    def forecast_prices(self, crop_name: str, yield_kg: float, quality_grade: str) -> PriceForecasting:
        """Complete price forecasting analysis"""
        
        current_price = self.get_current_market_price(crop_name)
        
        # Quality adjustment
        quality_multipliers = {
            "Grade A (Premium Export)": 1.2,
            "Grade B (Standard Export)": 1.0,
            "Grade C (Domestic Market)": 0.85,
            "Below Standard": 0.7
        }
        
        adjusted_price = current_price * quality_multipliers.get(quality_grade, 1.0)
        
        # Price predictions
        price_7d, price_30d, trend = self.predict_price_trends(crop_name, adjusted_price, yield_kg)
        
        # Market metrics
        supply_demand_ratio, volatility, seasonal_factor = self.calculate_market_metrics(yield_kg)
        
        # Revenue calculation
        revenue_estimate = yield_kg * adjusted_price
        
        # Confidence score based on data quality
        confidence_score = 0.8 if quality_grade.startswith("Grade") else 0.6
        
        return PriceForecasting(
            current_price=adjusted_price,
            predicted_price_7d=price_7d,
            predicted_price_30d=price_30d,
            market_trend=trend,
            seasonal_factor=seasonal_factor,
            supply_demand_ratio=supply_demand_ratio,
            price_volatility=volatility,
            revenue_estimate=revenue_estimate,
            confidence_score=confidence_score
        )