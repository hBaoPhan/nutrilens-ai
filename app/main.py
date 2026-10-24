# ai-service/app/main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from app.schemas import SegmentationResponse
from app.segmenter import FoodSegmentationEngine

app = FastAPI(
    title="AR NutriLens - Food Segmentation Microservice",
    description="Instance segmentation pipeline extracting 2D food component polygons for 3D depth raycasting",
    version="1.0.0"
)

default_weights = "weights/yolov8_vietfood_seg.pt" if os.path.exists("weights/yolov8_vietfood_seg.pt") else "yolov8n-seg.pt"
weights_path = os.getenv("MODEL_WEIGHTS_PATH", default_weights)
engine = FoodSegmentationEngine(weights_path=weights_path)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "HEALTHY", "service": "nutrilens-ai-segmentation"}

@app.post("/api/v1/vision/segment", response_model=SegmentationResponse, status_code=status.HTTP_200_OK)
async def segment_food(image: UploadFile = File(...)):
    # Validate MIME type to prevent malformed or malicious payload processing
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format: {image.content_type}. Only JPEG, PNG, and WebP are allowed."
        )

    content = await image.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image payload is empty"
        )

    items = engine.segment_image_bytes(content)

    return SegmentationResponse(
        status="SUCCESS",
        totalComponentsDetected=len(items),
        items=items
    )
