# ai-service/app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class NormalizedPoint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate in [0, 1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate in [0, 1]")

class SegmentedFoodItem(BaseModel):
    foodCode: str = Field(..., description="Standardized food code identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    polygonNormalized: List[List[float]] = Field(..., description="List of [x, y] coordinates forming 2D contour")
    boundingBox: List[float] = Field(..., description="[x_min, y_min, x_max, y_max]")

class SegmentationResponse(BaseModel):
    status: str = Field("SUCCESS", description="Operation status")
    totalComponentsDetected: int = Field(..., ge=0)
    items: List[SegmentedFoodItem]
    allDetected: Optional[List[SegmentedFoodItem]] = Field(
        default=None,
        description="All candidates detected by YOLO model including low confidence (< 0.40)"
    )
