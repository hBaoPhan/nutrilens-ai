# ai-service/app/segmenter.py
import os
import math
from typing import List, Tuple
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
    Supports multi-angle auto-orientation recovery for mobile/webcam rotated inputs.
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
                if os.path.exists("weights/yolov8_vietfood_seg.pt"):
                    target_weights = "weights/yolov8_vietfood_seg.pt"
                elif os.path.exists("yolov8_vietfood_seg.pt"):
                    target_weights = "yolov8_vietfood_seg.pt"
                else:
                    target_weights = "yolov8n-seg.pt"
            print(f"[NutriLens AI] Loading YOLO segmentation model from: {target_weights}")
            self.model = YOLO(target_weights)
            print("[NutriLens AI] YOLO model loaded successfully!")
        except Exception as e:
            print(f"[NutriLens AI] Warning: Could not initialize YOLO model: {e}")
            self.model = None

    def segment_image_bytes(
        self,
        image_bytes: bytes,
        conf_threshold: float = 0.40,
        scan_min_conf: float = 0.01,
        auto_orient: bool = True
    ) -> Tuple[List[SegmentedFoodItem], List[SegmentedFoodItem]]:
        """
        Runs YOLO segmentation inference on image bytes.
        Includes automatic orientation recovery for rotated/mirrored camera feeds.
        """
        if self.model is not None:
            try:
                import cv2
                import numpy as np

                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return [], []

                orig_h, orig_w = img.shape[:2]
                effective_scan_conf = min(scan_min_conf, conf_threshold)

                # 1. First pass: run inference at normal 0 degree orientation
                results = self.model(img, conf=effective_scan_conf, verbose=False)[0]
                all_detected = self._extract_items_from_results(results, orig_w, orig_h)
                qualified_items = [it for it in all_detected if it.confidence >= conf_threshold]

                # 2. Multi-Angle Orientation Recovery:
                # If no food detected with conf >= conf_threshold at 0 deg,
                # test 90 deg CW, 270 deg CW, 180 deg, and vertical flip
                if len(qualified_items) == 0 and auto_orient:
                    orientations = [
                        ("rot_90", cv2.ROTATE_90_CLOCKWISE),
                        ("rot_270", cv2.ROTATE_90_COUNTERCLOCKWISE),
                        ("rot_180", cv2.ROTATE_180),
                        ("flip_v", None)
                    ]

                    best_rot_qual = []
                    best_rot_all = []
                    best_angle_name = None

                    for name, rot_code in orientations:
                        if name == "flip_v":
                            rot_img = cv2.flip(img, 0)
                        else:
                            rot_img = cv2.rotate(img, rot_code)

                        rot_h, rot_w = rot_img.shape[:2]
                        rot_res = self.model(rot_img, conf=effective_scan_conf, verbose=False)[0]
                        rot_raw_items = self._extract_items_from_results(rot_res, rot_w, rot_h)

                        # Re-map bounding box and mask polygon coordinates back to original image
                        mapped_items = self._remap_items_to_original(rot_raw_items, name, orig_w, orig_h, rot_w, rot_h)
                        rot_qual = [it for it in mapped_items if it.confidence >= conf_threshold]

                        if len(rot_qual) > len(best_rot_qual):
                            best_rot_qual = rot_qual
                            best_rot_all = mapped_items
                            best_angle_name = name

                    if len(best_rot_qual) > 0:
                        print(f"[NutriLens AI] 🔄 Auto-Orientation Applied: Successfully recovered {len(best_rot_qual)} food item(s) via {best_angle_name}")
                        all_detected = best_rot_all
                        qualified_items = best_rot_qual

                # Sort by confidence descending
                all_detected.sort(key=lambda x: x.confidence, reverse=True)
                qualified_items.sort(key=lambda x: x.confidence, reverse=True)

                # Fallback to atomic ingredients only if enabled and no items qualified
                enable_atomic_fallback = os.getenv("ENABLE_ATOMIC_FALLBACK", "false").lower() == "true"
                if len(qualified_items) == 0 and enable_atomic_fallback:
                    atomic_items = self._detect_atomic_ingredients(img)
                    if atomic_items:
                        qualified_items.extend(atomic_items)
                        all_detected.extend(atomic_items)

                return qualified_items, all_detected
            except Exception as ex:
                print(f"[NutriLens AI] Inference runtime error: {ex}")
                return [], []

        # If model is not loaded, check if simulation fallback is explicitly enabled
        if os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() == "true":
            mock_items = self._fallback_simulation()
            return mock_items, mock_items

        return [], []

    def _extract_items_from_results(self, results, w: int, h: int) -> List[SegmentedFoodItem]:
        items = []
        if results.boxes is not None and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                raw_class_name = self.model.names[int(box.cls[0])].upper()
                food_code = FOOD_MAPPING.get(raw_class_name, raw_class_name)
                confidence = round(float(box.conf[0]), 3)
                bbox = [round(c, 2) for c in box.xyxy[0].tolist()]

                polygon = []
                if results.masks is not None and i < len(results.masks.xyn):
                    mask_pts = results.masks.xyn[i]
                    if len(mask_pts) > 0:
                        polygon = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in mask_pts.tolist()]

                if not polygon:
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
        return items

    def _remap_items_to_original(self, items: List[SegmentedFoodItem], orientation: str, orig_w: int, orig_h: int, rot_w: int, rot_h: int) -> List[SegmentedFoodItem]:
        remapped = []
        for it in items:
            # Map normalized polygon points back to original image space
            new_poly = []
            for p in it.polygonNormalized:
                u_r, v_r = p[0], p[1]
                if orientation == "rot_90": # Clockwise 90
                    u = v_r
                    v = 1.0 - u_r
                elif orientation == "rot_270": # Counter-Clockwise 90
                    u = 1.0 - v_r
                    v = u_r
                elif orientation == "rot_180":
                    u = 1.0 - u_r
                    v = 1.0 - v_r
                elif orientation == "flip_v":
                    u = u_r
                    v = 1.0 - v_r
                else:
                    u, v = u_r, v_r
                new_poly.append([round(max(0.0, min(1.0, u)), 3), round(max(0.0, min(1.0, v)), 3)])

            # Compute new bounding box from remapped polygon
            all_x = [p[0] * orig_w for p in new_poly]
            all_y = [p[1] * orig_h for p in new_poly]
            new_bbox = [
                round(min(all_x), 2),
                round(min(all_y), 2),
                round(max(all_x), 2),
                round(max(all_y), 2)
            ]

            remapped.append(SegmentedFoodItem(
                foodCode=it.foodCode,
                confidence=it.confidence,
                polygonNormalized=new_poly,
                boundingBox=new_bbox
            ))
        return remapped

    def _detect_atomic_ingredients(self, img) -> List[SegmentedFoodItem]:
        try:
            import cv2
            import numpy as np

            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
