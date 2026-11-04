# ai-service/app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any

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
    itemCounts: Optional[dict] = Field(
        default_factory=dict,
        description="Summary counts of detected instances per food code, e.g. {'BOILED_EGG': 2}"
    )
    items: List[SegmentedFoodItem]
    allDetected: Optional[List[SegmentedFoodItem]] = Field(
        default=None,
        description="All candidates detected by YOLO model including low confidence (< 0.40)"
    )

# ==============================================================================
# RAG Meal Recommendation Schemas
# ==============================================================================

class UserContextDTO(BaseModel):
    userId: Optional[int] = None
    username: Optional[str] = "user1"
    fitnessGoal: str = "MAINTAIN"
    weightKg: float = 65.0
    heightCm: float = 170.0
    dailyCalorieGoal: int = 2000
    todayConsumedKcal: float = 0.0
    remainingKcalToday: float = 2000.0
    targetMealType: str = "DINNER"
    targetMealKcal: float = 600.0
    targetMealProteinG: float = 25.0
    alreadyEatenFoods: List[str] = []

class CandidateFoodDTO(BaseModel):
    foodCode: str
    displayName: str
    category: Optional[str] = "Dinner"
    foodType: Optional[str] = "COMPOSITE_DISH"
    densityGramsPerCm3: float = 1.0
    servingWeightG: float = 100.0
    kcalPer100g: float = 100.0
    proteinPer100g: float = 5.0
    carbPer100g: float = 10.0
    fatPer100g: float = 2.0
    emoji: Optional[str] = "🍲"
    iconType: Optional[str] = "salad"
    iconBgColor: Optional[str] = "#FFEFE6"
    iconColor: Optional[str] = "#F37340"

class RAGRecommendationRequest(BaseModel):
    userContext: UserContextDTO
    candidateFoods: List[CandidateFoodDTO]

class RAGRecommendedDish(BaseModel):
    foodCode: str
    displayName: str
    category: Optional[str] = "Dinner"
    foodType: Optional[str] = "COMPOSITE_DISH"
    densityGramsPerCm3: float = 1.0
    suggestedWeightG: float = 100.0
    suggestedVolumeCm3: float = 100.0
    estimatedKcal: float
    proteinG: float
    carbG: float
    fatG: float
    emoji: Optional[str] = "🍲"
    iconType: Optional[str] = "salad"
    iconBgColor: Optional[str] = "#FFEFE6"
    iconColor: Optional[str] = "#F37340"
    matchScore: float = 0.95
    suitabilityNote: str

class RAGRecommendationResponse(BaseModel):
    status: str = "SUCCESS"
    mealType: str
    targetMealKcal: float
    remainingKcalToday: float
    aiReasoningSummary: str
    modelSource: str = "AI_RAG_CLINICAL_SYNTHESIZER"
    recommendedDishes: List[RAGRecommendedDish]
