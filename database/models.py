from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime
from sqlalchemy.sql import func
from database.config import Base

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