# ai-service/app/main.py
import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from app.schemas import SegmentationResponse
from app.segmenter import FoodSegmentationEngine

# Configure structured logging for AI Service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [NutriLens-AI] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("nutrilens-ai")

app = FastAPI(
    title="AR NutriLens - Food Segmentation Microservice",
    description="Instance segmentation pipeline extracting 2D food component polygons for 3D depth raycasting",
    version="1.0.0"
)

default_weights = "weights/yolov8_vietfood_seg.pt" if os.path.exists("weights/yolov8_vietfood_seg.pt") else "yolov8n-seg.pt"
weights_path = os.getenv("MODEL_WEIGHTS_PATH", default_weights)
engine = FoodSegmentationEngine(weights_path=weights_path)

# Debug captures directory
DEBUG_CAPTURES_DIR = "debug_captures"
os.makedirs(DEBUG_CAPTURES_DIR, exist_ok=True)

# Artifact captures directory if on host
ARTIFACT_DIR = r"C:\Users\baoph\.gemini\antigravity\brain\fd8f0d58-fe89-4a9f-a991-bf55be9f40a3\class_test_images"

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    model_loaded = engine.model is not None
    return {
        "status": "HEALTHY",
        "service": "nutrilens-ai-segmentation",
        "modelLoaded": model_loaded,
        "weights": engine.weights_path
    }

@app.post("/api/v1/vision/segment", response_model=SegmentationResponse, status_code=status.HTTP_200_OK)
async def segment_food(
    image: UploadFile = File(...),
    min_conf: Optional[float] = Query(None, description="Optional minimum confidence threshold override (e.g. 0.40)"),
    include_all: bool = Query(False, description="If true, returns all detected candidates in items list")
):
    start_time = time.time()
    filename = image.filename or "unknown"
    content_type = image.content_type or "unknown"

    logger.info(f"📸 Incoming POST /api/v1/vision/segment | File: {filename} ({content_type})")

    # Validate MIME type to prevent malformed or malicious payload processing
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        logger.warning(f"❌ Rejected request: Unsupported image format '{image.content_type}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format: {image.content_type}. Only JPEG, PNG, and WebP are allowed."
        )

    content = await image.read()
    if len(content) == 0:
        logger.warning("❌ Rejected request: Uploaded image payload is empty")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image payload is empty"
        )

    size_kb = len(content) / 1024.0

    # Save exact received snapshot for live debugging & inspection
    try:
        debug_path = os.path.join(DEBUG_CAPTURES_DIR, "last_unity_capture.jpg")
        with open(debug_path, "wb") as f:
            f.write(content)
        if os.path.exists(ARTIFACT_DIR):
            artifact_snap_path = os.path.join(ARTIFACT_DIR, "last_unity_capture.jpg")
            with open(artifact_snap_path, "wb") as f:
                f.write(content)
    except Exception as save_ex:
        logger.warning(f"Could not persist debug snapshot: {save_ex}")

    # Confidence threshold configuration: default is 0.40 or environment override
    default_conf_thresh = float(os.getenv("YOLO_CONF_THRESHOLD", "0.40"))
    effective_conf_thresh = min_conf if min_conf is not None else default_conf_thresh
    scan_min_conf = float(os.getenv("YOLO_SCAN_MIN_CONF", "0.01"))

    # Perform segmentation with full scanning of all objects and multi-angle auto-orientation
    qualified_items, all_detected = engine.segment_image_bytes(
        content,
        conf_threshold=effective_conf_thresh,
        scan_min_conf=scan_min_conf,
        auto_orient=True
    )
    inference_ms = (time.time() - start_time) * 1000.0

    # Categorize detected objects
    low_conf_items = [it for it in all_detected if it.confidence < effective_conf_thresh]

    # COMPREHENSIVE LOGGING OF ALL DETECTED OBJECTS (INCLUDING CONF < 0.40)
    if len(all_detected) == 0:
        logger.info(
            f"🔍 Scan Result ({size_kb:.1f} KB | {inference_ms:.1f}ms): "
            f"0 objects detected by YOLO model (tested 0°, 90°, 180°, 270°, and vertical flip)"
        )
    else:
        logger.info(
            f"🎯 Scan Result ({size_kb:.1f} KB | {inference_ms:.1f}ms): "
            f"Total {len(all_detected)} object(s) detected "
            f"[{len(qualified_items)} qualified (>= {effective_conf_thresh * 100:.0f}%), "
            f"{len(low_conf_items)} low confidence (< {effective_conf_thresh * 100:.0f}%)]"
        )
        for idx, it in enumerate(all_detected, start=1):
            mask_pts = len(it.polygonNormalized) if it.polygonNormalized else 0
            if it.confidence >= effective_conf_thresh:
                status_tag = f"✅ [QUALIFIED >= {effective_conf_thresh * 100:.0f}%]"
            else:
                status_tag = f"⚠️ [LOW_CONF  < {effective_conf_thresh * 100:.0f}%]"

            logger.info(
                f"   [#{idx:02d}] {status_tag} FoodCode: {it.foodCode:<22} | "
                f"Conf: {it.confidence * 100:5.1f}% | BBox: {it.boundingBox} | Mask: {mask_pts} pts"
            )

    # If include_all is requested, return all detected items in items field
    output_items = all_detected if include_all else qualified_items

    return SegmentationResponse(
        status="SUCCESS",
        totalComponentsDetected=len(output_items),
        items=output_items,
        allDetected=all_detected
    )
