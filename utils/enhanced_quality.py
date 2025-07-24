import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class ExportQuality:
    """Data class for export quality metrics"""
    fruit_count: int
    avg_size: str
    size_distribution: Dict[str, int]
    ripeness: str
    ripeness_score: float
    defect_rate: float
    grade: str
    export_ready_percentage: float
    harvest_date: str
    shelf_life_days: int

class ExportQualityAssessment:
    """Enhanced export quality assessment system"""
    
    def __init__(self):
        self.quality_standards = {
            "Grade A": {"min_size": "Medium", "max_defect_rate": 0.05, "min_ripeness": 0.7},
            "Grade B": {"min_size": "Small", "max_defect_rate": 0.15, "min_ripeness": 0.5},
            "Grade C": {"min_size": "Small", "max_defect_rate": 0.30, "min_ripeness": 0.3}
        }
    
    def analyze_fruit_size(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[str, Dict[str, int]]:
        """Enhanced fruit size analysis with distribution"""
        if not bboxes:
            return "Unknown", {}
        
        sizes = []
        size_distribution = {"Small": 0, "Medium": 0, "Large": 0}
        
        for box in bboxes:
            x1, y1, x2, y2 = box
            area = abs((x2 - x1) * (y2 - y1))
            
            if area < 1200:
                size = "Small"
            elif area < 2800:
                size = "Medium"
            else:
                size = "Large"
            
            sizes.append(size)
            size_distribution[size] += 1
        
        # Determine average size
        avg_size = max(set(sizes), key=sizes.count) if sizes else "Unknown"
        return avg_size, size_distribution
    
    def analyze_ripeness(self, image_path: str) -> Tuple[str, float]:
        """Enhanced ripeness analysis with scoring"""
        try:
            if not os.path.exists(image_path):
                print(f"Image path does not exist: {image_path}")
                return "Unknown", 0.0
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return "Unknown", 0.0
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Enhanced color analysis for different ripeness stages
            # Red range for ripe fruits
            red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Yellow/orange range for semi-ripe
            yellow_mask = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([35, 255, 255]))
            
            # Green range for unripe
            green_mask = cv2.inRange(hsv, np.array([40, 70, 50]), np.array([80, 255, 255]))
            
            total_pixels = img.shape[0] * img.shape[1]
            red_ratio = cv2.countNonZero(red_mask) / total_pixels
            yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
            green_ratio = cv2.countNonZero(green_mask) / total_pixels
            
            # Calculate ripeness score (0-1)
            ripeness_score = red_ratio * 1.0 + yellow_ratio * 0.6 + green_ratio * 0.2
            
            # Determine ripeness category
            if ripeness_score >= 0.7:
                ripeness = "Fully Ripe"
            elif ripeness_score >= 0.5:
                ripeness = "Ripe"
            elif ripeness_score >= 0.3:
                ripeness = "Semi-Ripe"
            else:
                ripeness = "Unripe"
            
            return ripeness, min(ripeness_score, 1.0)
            
        except Exception as e:
            print(f"Error analyzing ripeness: {e}")
            return "Unknown", 0.0
    
    def detect_defects(self, image_path: str, detections: List[Dict]) -> float:
        """Detect defects and calculate defect rate"""
        try:
            # Placeholder for defect detection algorithm
            # In a real implementation, this would use computer vision to detect:
            # - Bruises, spots, discoloration
            # - Shape irregularities
            # - Size inconsistencies
            
            defect_indicators = 0
            total_fruits = len(detections)
            
            if total_fruits == 0:
                return 0.0
            
            # Simple defect estimation based on confidence scores
            low_confidence_count = sum(1 for d in detections if d.get('score', 1.0) < 0.7)
            defect_rate = low_confidence_count / total_fruits
            
            return min(defect_rate, 1.0)
            
        except Exception as e:
            print(f"Error detecting defects: {e}")
            return 0.1  # Default defect rate
    
    def calculate_export_grade(self, size: str, defect_rate: float, ripeness_score: float) -> Tuple[str, float]:
        """Calculate export grade and export readiness percentage"""
        
        size_scores = {"Large": 1.0, "Medium": 0.8, "Small": 0.6}
        size_score = size_scores.get(size, 0.5)
        
        # Calculate overall quality score
        quality_score = (
            size_score * 0.3 +
            (1 - defect_rate) * 0.4 +
            ripeness_score * 0.3
        )
        
        # Determine grade
        if quality_score >= 0.85:
            grade = "Grade A (Premium Export)"
        elif quality_score >= 0.70:
            grade = "Grade B (Standard Export)"
        elif quality_score >= 0.55:
            grade = "Grade C (Domestic Market)"
        else:
            grade = "Below Standard"
        
        export_ready_percentage = quality_score * 100
        
        return grade, export_ready_percentage
    
    def estimate_harvest_date(self, ripeness: str, ripeness_score: float) -> Tuple[str, int]:
        """Enhanced harvest date estimation with shelf life"""
        
        # Days until optimal harvest based on ripeness
        days_mapping = {
            "Fully Ripe": (1, 3),      # (harvest_days, shelf_life)
            "Ripe": (3, 7),
            "Semi-Ripe": (7, 14),
            "Unripe": (14, 21),
            "Unknown": (7, 10)
        }
        
        harvest_days, shelf_life = days_mapping.get(ripeness, (7, 10))
        
        # Adjust based on ripeness score
        if ripeness_score > 0.8:
            harvest_days = max(1, harvest_days - 2)
        elif ripeness_score < 0.3:
            harvest_days += 3
        
        harvest_date = (datetime.now() + timedelta(days=harvest_days)).strftime('%Y-%m-%d')
        
        return harvest_date, shelf_life
    
    def assess_export_quality(self, detections: List[Dict], image_path: str) -> ExportQuality:
        """Complete export quality assessment"""
        
        if not detections:
            return ExportQuality(
                fruit_count=0,
                avg_size="Unknown",
                size_distribution={},
                ripeness="Unknown",
                ripeness_score=0.0,
                defect_rate=0.0,
                grade="No Data",
                export_ready_percentage=0.0,
                harvest_date="Unknown",
                shelf_life_days=0
            )
        
        # Extract bounding boxes
        bboxes = [(d["box"][0], d["box"][1], d["box"][2], d["box"][3]) for d in detections]
        
        # Analyze components
        avg_size, size_distribution = self.analyze_fruit_size(bboxes)
        ripeness, ripeness_score = self.analyze_ripeness(image_path)
        defect_rate = self.detect_defects(image_path, detections)
        grade, export_ready_percentage = self.calculate_export_grade(avg_size, defect_rate, ripeness_score)
        harvest_date, shelf_life = self.estimate_harvest_date(ripeness, ripeness_score)
        
        return ExportQuality(
            fruit_count=len(detections),
            avg_size=avg_size,
            size_distribution=size_distribution,
            ripeness=ripeness,
            ripeness_score=ripeness_score,
            defect_rate=defect_rate,
            grade=grade,
            export_ready_percentage=export_ready_percentage,
            harvest_date=harvest_date,
            shelf_life_days=shelf_life
        )