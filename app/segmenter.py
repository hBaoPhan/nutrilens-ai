# ai-service/app/segmenter.py
import os
import math
from typing import List
from app.schemas import SegmentedFoodItem

# Mapping detected object/food classes to standardized NutriLens food codes
FOOD_MAPPING = {
    # COCO Fruits & Foods
    "APPLE": "APPLE",
    "BANANA": "BANANA",
    "ORANGE": "ORANGE",
    "BROCCOLI": "BROCCOLI",
    "CARROT": "CARROT",
    "SANDWICH": "AVOCADO_TOAST",
    "PIZZA": "PIZZA",
    "HOT DOG": "PORK_GRILLED_RIB",
    "CAKE": "GREEK_YOGURT",
    "DONUT": "GREEK_YOGURT",
    "BOWL": "RICE_WHITE_COOKED",
    "CUP": "GREEK_YOGURT",
    
    # Custom Vietnamese Food Classes
    "RICE_WHITE_COOKED": "RICE_WHITE_COOKED",
    "BROKEN_RICE": "BROKEN_RICE",
    "PORK_GRILLED_RIB": "PORK_GRILLED_RIB",
    "BEEF_PHO": "BEEF_PHO",
    "BUN_CHA_HA_NOI": "BUN_CHA_HA_NOI",
    "BANH_CUON": "BANH_CUON",
    "BANH_GIO": "BANH_GIO",
    "BANH_MI": "BANH_MI",
    "BUN_BO_HUE": "BUN_BO_HUE",
    "BUN_DAU_MAM_TOM": "BUN_DAU_MAM_TOM",
    "BUN_THIT_NUONG": "BUN_THIT_NUONG",
    "CHAO_LONG": "CHAO_LONG",
    "COM_DEP": "COM_DEP",
    "COM_RANG": "COM_RANG",
    "GOI_CUON": "GOI_CUON",
    "XOI": "XOI",
    "SALAD_EGG": "SALAD_EGG",
    "SALMON_STEAK": "SALMON_STEAK",
    "GRILLED_CHICKEN_BREAST": "GRILLED_CHICKEN_BREAST",
    "CHICKEN_SALAD": "CHICKEN_SALAD",
    "CANH_CHUA_CA_HOI": "CANH_CHUA_CA_HOI",
    "BEEF_QUINOA_BOWL": "BEEF_QUINOA_BOWL",
    "OATMEAL_BERRY": "OATMEAL_BERRY",

    # Atomic Food Ingredients
    "BOILED_EGG": "BOILED_EGG",
    "BOILED EGG": "BOILED_EGG",
    "EGG": "BOILED_EGG",
    "CHICKEN_BREAST": "CHICKEN_BREAST",
    "SWEET_POTATO": "SWEET_POTATO",
    "TOFU_STEAMED": "TOFU_STEAMED",
    "SHRIMP_BOILED": "SHRIMP_BOILED",
    "CUCUMBER": "CUCUMBER",
    "RICE_WHITE": "RICE_WHITE",
    "PORK_GRILLED": "PORK_GRILLED",

    # Dataset Label Names to Standard Codes
    "STEAMED RICE ROLLS": "BANH_CUON",
    "VIETNAMESE PYRAMID RICE DUMPLING": "BANH_GIO",
    "GRILLED PORK BANH MI": "BANH_MI",
    "HUE-STYLE SPICY BEEF NOODLE SOUP": "BUN_BO_HUE",
    "GRILLED PORK WITH VERMICELLI": "BUN_CHA_HA_NOI",
    "VERMICELLI WITH TOFU": "BUN_DAU_MAM_TOM",
    "GRILLED PORK VERMICELLI": "BUN_THIT_NUONG",
    "PORK OFFAL PORRIDGE": "CHAO_LONG",
    "FLATTENED GREEN RICE": "COM_DEP",
    "BROKEN RICE": "BROKEN_RICE",
    "FRIED RICE": "COM_RANG",
    "BEEF PHO": "BEEF_PHO",
    "SPRING ROLLS": "GOI_CUON",
    "STICKY RICE": "XOI"
}

class FoodSegmentationEngine:
    """
    Real-world Instance Segmentation inference engine.
    Loads YOLOv8 segmentation weights (custom or pretrained yolov8n-seg.pt)
    and extracts actual 2D polygon masks and bounding boxes from camera images.
    """
    def __init__(self, weights_path: str = None):
        self.weights_path = weights_path
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            target_weights = self.weights_path
            if not target_weights or not os.path.exists(target_weights):
                target_weights = "yolov8n-seg.pt" # Auto-downloads official Ultralytics weights
            print(f"[NutriLens AI] Loading YOLO segmentation model from: {target_weights}")
            self.model = YOLO(target_weights)
            print("[NutriLens AI] YOLO model loaded successfully!")
        except Exception as e:
            print(f"[NutriLens AI] Warning: Could not initialize YOLO model: {e}")
            self.model = None

    def segment_image_bytes(self, image_bytes: bytes) -> List[SegmentedFoodItem]:
        # If model is loaded, run real-time inference on actual camera pixels
        if self.model is not None:
            try:
                import cv2
                import numpy as np

                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return []

                conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.40"))
                results = self.model(img, conf=conf_threshold)[0]
                items = []

                if results.boxes is not None and len(results.boxes) > 0:
                    for i, box in enumerate(results.boxes):
                        raw_class_name = self.model.names[int(box.cls[0])].upper()
                        # Map detected object to standard NutriLens foodCode
                        food_code = FOOD_MAPPING.get(raw_class_name, raw_class_name)
                        confidence = round(float(box.conf[0]), 3)
                        bbox = [round(c, 2) for c in box.xyxy[0].tolist()]

                        polygon = []
                        if results.masks is not None and i < len(results.masks.xyn):
                            mask_pts = results.masks.xyn[i]
                            if len(mask_pts) > 0:
                                polygon = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in mask_pts.tolist()]

                        # If polygon mask was not extracted, fallback to normalized bbox box
                        if not polygon:
                            h, w = img.shape[:2]
                            polygon = [
                                [bbox[0] / w, bbox[1] / h],
                                [bbox[2] / w, bbox[1] / h],
                                [bbox[2] / w, bbox[3] / h],
                                [bbox[0] / w, bbox[3] / h],
                                [bbox[0] / w, bbox[1] / h]
                            ]

                        items.append(SegmentedFoodItem(
                            foodCode=food_code,
                            confidence=confidence,
                            polygonNormalized=polygon,
                            boundingBox=bbox
                        ))

                # Fallback to atomic ingredients (disabled by default to prevent false positives like auto-egg detection)
                enable_atomic_fallback = os.getenv("ENABLE_ATOMIC_FALLBACK", "false").lower() == "true"
                if len(items) == 0 and enable_atomic_fallback:
                    items = self._detect_atomic_ingredients(img)

                # Return true detected items (or empty list if no food detected)
                return items
            except Exception as ex:
                print(f"[NutriLens AI] Inference runtime error: {ex}")

        # If model is not loaded, check if simulation fallback is explicitly enabled via env var
        if os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() == "true":
            return self._fallback_simulation()

        return []

    def _detect_atomic_ingredients(self, img) -> List[SegmentedFoodItem]:
        try:
            import cv2
            import numpy as np

            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Egg shell tones: Range 1: Warm beige/brown egg shell, Range 2: White egg shell
            mask1 = cv2.inRange(hsv, np.array([5, 15, 100]), np.array([32, 210, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 45, 255]))
            mask = cv2.bitwise_or(mask1, mask2)

            k_size = max(5, int(min(h, w) * 0.02)) | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            egg_pixels = cv2.countNonZero(mask)
            ratio = egg_pixels / float(h * w)

            items = []
            if ratio > 0.12:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) > (h * w * 0.02)]
                if valid_contours:
                    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
                    main_c = valid_contours[0]
                    x, y, bw, bh = cv2.boundingRect(main_c)

                    epsilon = 0.02 * cv2.arcLength(main_c, True)
                    approx = cv2.approxPolyDP(main_c, epsilon, True)
                    poly = [[round(float(p[0][0]) / w, 3), round(float(p[0][1]) / h, 3)] for p in approx]
                    if len(poly) < 3:
                        poly = [[round(x / w, 3), round(y / h, 3)],
                                [round((x + bw) / w, 3), round(y / h, 3)],
                                [round((x + bw) / w, 3), round((y + bh) / h, 3)],
                                [round(x / w, 3), round((y + bh) / h, 3)]]

                    items.append(SegmentedFoodItem(
                        foodCode="BOILED_EGG",
                        confidence=0.915,
                        polygonNormalized=poly,
                        boundingBox=[round(float(x), 2), round(float(y), 2), round(float(x + bw), 2), round(float(y + bh), 2)]
                    ))
            return items
        except Exception as ex:
            print(f"[NutriLens AI] Atomic ingredient detection error: {ex}")
            return []

    def _fallback_simulation(self) -> List[SegmentedFoodItem]:
        return [
            SegmentedFoodItem(
                foodCode="RICE_WHITE_COOKED",
                confidence=0.965,
                polygonNormalized=[
                    [0.25, 0.30], [0.55, 0.28], [0.60, 0.58], [0.30, 0.62], [0.25, 0.30]
                ],
                boundingBox=[0.25, 0.28, 0.60, 0.62]
            ),
            SegmentedFoodItem(
                foodCode="PORK_GRILLED_RIB",
                confidence=0.942,
                polygonNormalized=[
                    [0.62, 0.35], [0.85, 0.38], [0.82, 0.70], [0.58, 0.68], [0.62, 0.35]
                ],
                boundingBox=[0.58, 0.35, 0.85, 0.70]
            )
        ]
