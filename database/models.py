from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from database.config import Base
from pydantic import BaseModel
from typing import List, Optional


"""
Author: Huong TA
Date: 2025-07-21
Description: Crate "image_detections" table in DB
"""
class ImageDetection(Base):
    __tablename__ = "image_detections"

    id = Column(Integer, primary_key=True, index=True)
    # image_id = Column(String, unique=True, nullable=False)
    image_id = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    detection_time = Column(Float)
    confidence = Column(Float)

    # Detection and correction data
    model_predictions = Column(JSON)  # Original model boxes
    user_annotations = Column(JSON)   # Corrected boxes

    num_objects_model = Column(Integer)
    num_objects_corrected = Column(Integer)

    annotated_by_user = Column(Boolean, default=False)
    annotator_id = Column(String, nullable=True)
    annotation_date = Column(DateTime, server_default=func.now())

# ----------------------------
# Detection DB endpoints
# ----------------------------
class DetectionIn(BaseModel):
    image_id: str
    image_path: str
    model_name: str
    detection_time: float
    confidence: float
    model_predictions: List[dict]
    user_annotations: Optional[List[dict]] = []
    num_objects_model: int
    num_objects_corrected: Optional[int] = 0
    annotated_by_user: bool = False
    annotator_id: Optional[str] = None

# Define request model
class SMSRequest(BaseModel):
    username: str
    file_name: str
    yield_kg: float
    phone_number: str   

# --- Pydantic Models for Authentication ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str # "admin", "annotator", "viewer"

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str

class UserPayload(BaseModel):
    username: str
    role: str
