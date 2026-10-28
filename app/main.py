# ai-service/app/main.py
import os
import time
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, status
from fastapi.responses import FileResponse
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
    description="Instance segmentation pipeline with in-depth image diagnostics, auto-orientation & debug capture exports",
    version="1.1.0"
)

default_weights = "weights/yolov8_vietfood_seg.pt" if os.path.exists("weights/yolov8_vietfood_seg.pt") else "yolov8n-seg.pt"
weights_path = os.getenv("MODEL_WEIGHTS_PATH", default_weights)
engine = FoodSegmentationEngine(weights_path=weights_path)

# Brain Artifact directory for Markdown embedding
ARTIFACT_DIR = r"C:\Users\baoph\.gemini\antigravity\brain\fd8f0d58-fe89-4a9f-a991-bf55be9f40a3\unity_captures"

# Cache last diagnostic info in memory for instant inspection
last_diagnostic_cache = {}


def get_debug_capture_dirs() -> List[str]:
    """
    Returns all valid target directories for saving debug captures.
    Guarantees saving to both root debug_captures and ai-service/debug_captures
    whether running inside Docker or directly on the host machine.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))  # /app/app or .../ai-service/app
    candidates = [
        # Inside Docker mount points
        "/app/debug_captures",
        "/app/ai_debug_captures",
        # Local development relative paths
        os.path.abspath("debug_captures"),
        os.path.abspath("ai-service/debug_captures"),
        os.path.abspath(os.path.join(base_dir, "..", "debug_captures")),
        os.path.abspath(os.path.join(base_dir, "..", "..", "debug_captures")),
        # Conversation Artifact directory
        ARTIFACT_DIR
    ]
    valid_dirs = []
    seen = set()
    for d in candidates:
        if d and d not in seen:
            seen.add(d)
            try:
                os.makedirs(d, exist_ok=True)
                if os.path.isdir(d) and os.access(d, os.W_OK):
                    valid_dirs.append(d)
            except Exception:
                pass
    return valid_dirs


def save_raw_debug_captures(content: bytes, timestamp_str: str) -> List[str]:
    """Saves raw incoming JPEG bytes to all debug directories."""
    saved_paths = []
    for d in get_debug_capture_dirs():
        p_ts = os.path.join(d, f"capture_{timestamp_str}.jpg")
        p_last = os.path.join(d, "last_unity_capture.jpg")
        for p in [p_ts, p_last]:
            try:
                with open(p, "wb") as f:
                    f.write(content)
                saved_paths.append(p)
            except Exception as e:
                logger.debug(f"Failed to write raw capture to {p}: {e}")
    return saved_paths


def draw_prediction_overlay(cv_img, all_detected, conf_threshold: float):
    """
    Draws bounding boxes, polygon masks, and confidence badges onto the image
    for instant visual inspection in debug_captures.
    """
    import cv2
    import numpy as np

    annotated = cv_img.copy()
    h, w = annotated.shape[:2]
    mask_layer = annotated.copy()

    # Distinguishable color palette (BGR)
    palette = [
        (46, 204, 113),   # Emerald Green
        (52, 152, 219),   # Sky Blue
        (155, 89, 182),   # Amethyst Purple
        (241, 196, 15),   # Sun Yellow
        (230, 126, 34),   # Orange
        (231, 76, 60),    # Coral Red
        (26, 188, 156),   # Turquoise
        (243, 156, 18)    # Amber
    ]

    has_mask = False
    for idx, it in enumerate(all_detected):
        color = palette[idx % len(palette)]
        is_qualified = it.confidence >= conf_threshold

        # 1. Fill polygon segmentation mask if present
        if it.polygonNormalized and len(it.polygonNormalized) >= 3:
            pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in it.polygonNormalized], np.int32)
            cv2.fillPoly(mask_layer, [pts], color)
            has_mask = True

        # 2. Draw bounding box rectangle
        if it.boundingBox and len(it.boundingBox) == 4:
            x1, y1, x2, y2 = [int(v) for v in it.boundingBox]
            box_color = color if is_qualified else (120, 120, 120)
            thickness = 2 if is_qualified else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # 3. Label text pill badge
            conf_pct = it.confidence * 100.0
            label = f"{it.foodCode} {conf_pct:.1f}%"
            font_scale = 0.52
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            badge_x2 = min(w, x1 + tw + 10)

            # Draw background badge rectangle
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
            text_color = (0, 0, 0) if is_qualified else (255, 255, 255)
            cv2.putText(
                annotated, label,
                (x1 + 4, badge_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
            )

    # Blend semi-transparent segmentation masks
    if has_mask:
        cv2.addWeighted(mask_layer, 0.35, annotated, 0.65, 0, annotated)

    return annotated


def save_prediction_debug_captures(annotated_img, timestamp_str: str) -> List[str]:
    """Encodes and saves annotated prediction image to all debug directories."""
    import cv2
    saved_paths = []
    for d in get_debug_capture_dirs():
        p_ts = os.path.join(d, f"pred_{timestamp_str}.jpg")
        p_last = os.path.join(d, "last_unity_prediction.jpg")
        for p in [p_ts, p_last]:
            try:
                cv2.imwrite(p, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved_paths.append(p)
            except Exception as e:
                logger.debug(f"Failed to write prediction image to {p}: {e}")
    return saved_paths


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    model_loaded = engine.model is not None
    return {
        "status": "HEALTHY",
        "service": "nutrilens-ai-segmentation",
        "modelLoaded": model_loaded,
        "weights": engine.weights_path,
        "debugCaptureDirs": get_debug_capture_dirs()
    }


@app.get("/api/v1/vision/last-capture", status_code=status.HTTP_200_OK)
async def get_last_capture():
    for d in get_debug_capture_dirs():
        last_file = os.path.join(d, "last_unity_capture.jpg")
        if os.path.exists(last_file):
            return FileResponse(last_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No capture received yet from Unity client")


@app.get("/api/v1/vision/last-prediction", status_code=status.HTTP_200_OK)
async def get_last_prediction():
    for d in get_debug_capture_dirs():
        last_file = os.path.join(d, "last_unity_prediction.jpg")
        if os.path.exists(last_file):
            return FileResponse(last_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No prediction image generated yet")


@app.get("/api/v1/vision/diagnose-last", status_code=status.HTTP_200_OK)
async def get_last_diagnostic():
    if not last_diagnostic_cache:
        raise HTTPException(status_code=404, detail="No capture diagnostic available yet")
    return last_diagnostic_cache


@app.post("/api/v1/vision/segment", response_model=SegmentationResponse, status_code=status.HTTP_200_OK)
async def segment_food(
    image: UploadFile = File(...),
    min_conf: Optional[float] = Query(None, description="Optional minimum confidence threshold override (e.g. 0.40)"),
    include_all: bool = Query(False, description="If true, returns all detected candidates in items list")
):
    start_time = time.time()
    filename = image.filename or "unknown"
    content_type = image.content_type or "unknown"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"📸 Incoming POST /api/v1/vision/segment | File: {filename} ({content_type})")

    # Validate MIME type
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

    # 1. SAVE RAW IMAGE IMMEDIATELY TO DEBUG_CAPTURES FOLDERS
    saved_raw = save_raw_debug_captures(content, timestamp_str)
    if saved_raw:
        logger.info(f"💾 Raw capture saved ({size_kb:.1f} KB): {os.path.basename(saved_raw[0])} -> {len(saved_raw)} locations")

    # 2. RUN IN-DEPTH COMPUTER VISION DIAGNOSTIC ON CAPTURED FRAME
    cv_img = None
    try:
        import cv2
        import numpy as np

        nparr = np.frombuffer(content, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv_img is not None:
            img_h, img_w = cv_img.shape[:2]
            aspect_ratio = img_w / float(img_h)
            orientation_type = "Landscape" if aspect_ratio > 1.05 else ("Portrait" if aspect_ratio < 0.95 else "Square")

            # Sharpness / Blur analysis (Laplacian Variance)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_status = "CRITICAL_BLUR" if laplacian_var < 35 else ("MILD_BLUR" if laplacian_var < 90 else "SHARP")

            # Brightness & Contrast analysis
            mean_brightness = float(np.mean(gray))
            contrast_std = float(np.std(gray))
            lighting_status = "TOO_DARK" if mean_brightness < 50 else ("TOO_BRIGHT" if mean_brightness > 215 else "NORMAL")

            # Color balance (BGR mean)
            mean_b, mean_g, mean_r = [float(v) for v in np.mean(cv_img, axis=(0, 1))]

            logger.info("═════════════════════════════════════════════════════════════════════")
            logger.info(f"🔬 CAMERA FRAME DIAGNOSTIC | Dim: {img_w}x{img_h} ({orientation_type}, aspect: {aspect_ratio:.2f})")
            logger.info(f"   • Blur Score (Laplacian Var): {laplacian_var:.1f} [{blur_status}]")
            logger.info(f"   • Brightness: {mean_brightness:.1f}/255 [{lighting_status}] | Contrast Std: {contrast_std:.1f}")
            logger.info(f"   • Channels Mean -> B: {mean_b:.1f}, G: {mean_g:.1f}, R: {mean_r:.1f}")

            # Test all 4 angles with YOLO to identify if orientation was wrong
            if engine.model is not None:
                angle_tests = {}
                test_angles = [
                    ("0_DEG", cv_img),
                    ("90_DEG_CW", cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)),
                    ("180_DEG", cv2.rotate(cv_img, cv2.ROTATE_180)),
                    ("270_DEG_CW", cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                    ("FLIP_V", cv2.flip(cv_img, 0))
                ]
                for aname, aimg in test_angles:
                    ares = engine.model(aimg, conf=0.005, verbose=False)[0]
                    atop = []
                    if ares.boxes is not None and len(ares.boxes) > 0:
                        for b in ares.boxes[:3]:
                            cname = engine.model.names[int(b.cls[0])]
                            cconf = float(b.conf[0])
                            atop.append(f"{cname} ({cconf*100:.1f}%)")
                    angle_tests[aname] = atop or ["None"]

                logger.info("   • Multi-Angle Predictions:")
                for aname, atop in angle_tests.items():
                    logger.info(f"     └─ {aname:<12}: {', '.join(atop)}")
            logger.info("═════════════════════════════════════════════════════════════════════")

            last_diagnostic_cache.update({
                "timestamp": timestamp_str,
                "resolution": f"{img_w}x{img_h}",
                "aspectRatio": round(aspect_ratio, 2),
                "orientation": orientation_type,
                "blurScore": round(laplacian_var, 1),
                "blurStatus": blur_status,
                "brightness": round(mean_brightness, 1),
                "lightingStatus": lighting_status,
                "contrast": round(contrast_std, 1),
                "angleTests": angle_tests if engine.model is not None else {}
            })
    except Exception as diag_ex:
        logger.warning(f"Diagnostic error: {diag_ex}")

    # 3. PERFORM STANDARD SEGMENTATION & INFERENCE
    default_conf_thresh = float(os.getenv("YOLO_CONF_THRESHOLD", "0.40"))
    effective_conf_thresh = min_conf if min_conf is not None else default_conf_thresh
    scan_min_conf = float(os.getenv("YOLO_SCAN_MIN_CONF", "0.01"))

    qualified_items, all_detected = engine.segment_image_bytes(
        content,
        conf_threshold=effective_conf_thresh,
        scan_min_conf=scan_min_conf,
        auto_orient=True
    )
    inference_ms = (time.time() - start_time) * 1000.0

    low_conf_items = [it for it in all_detected if it.confidence < effective_conf_thresh]

    if len(all_detected) == 0:
        logger.info(
            f"🔍 Scan Result ({size_kb:.1f} KB | {inference_ms:.1f}ms): "
            f"0 objects detected by YOLO model across all orientations"
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
            status_tag = f"✅ [QUALIFIED >= {effective_conf_thresh * 100:.0f}%]" if it.confidence >= effective_conf_thresh else f"⚠️ [LOW_CONF  < {effective_conf_thresh * 100:.0f}%]"
            logger.info(
                f"   [#{idx:02d}] {status_tag} FoodCode: {it.foodCode:<22} | "
                f"Conf: {it.confidence * 100:5.1f}% | BBox: {it.boundingBox} | Mask: {mask_pts} pts"
            )

    # 4. GENERATE AND SAVE ANNOTATED VISUAL PREDICTION OVERLAY
    if cv_img is not None:
        try:
            annotated_img = draw_prediction_overlay(cv_img, all_detected, effective_conf_thresh)
            saved_pred = save_prediction_debug_captures(annotated_img, timestamp_str)
            if saved_pred:
                logger.info(f"🎨 Visual prediction overlay saved: pred_{timestamp_str}.jpg -> {len(saved_pred)} locations")
        except Exception as draw_ex:
            logger.warning(f"Could not generate prediction overlay image: {draw_ex}")

    output_items = all_detected if include_all else qualified_items

    return SegmentationResponse(
        status="SUCCESS",
        totalComponentsDetected=len(output_items),
        items=output_items,
        allDetected=all_detected
    )
