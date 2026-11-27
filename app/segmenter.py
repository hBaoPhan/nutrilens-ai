# ai-service/app/segmenter.py
import os
import logging
from typing import List, Optional, Tuple
from app.schemas import SegmentedFoodItem

logger = logging.getLogger("nutrilens-ai")

# Standardized mapping from YOLO class labels to canonical NutriLens food codes
FOOD_MAPPING = {
    # Fruits & Yogurt Classes
    "APPLE": "APPLE",
    "BANANA": "BANANA",
    "ORANGE": "ORANGE",
    "COCONUT": "COCONUT",
    "WATERMELON": "WATERMELON",
    "SUA_CHUA": "SUA_CHUA",
    "YOGURT": "SUA_CHUA",
    "GREEK_YOGURT": "SUA_CHUA",
    "BROCCOLI": "BROCCOLI",
    "CARROT": "CARROT",
    "PIZZA": "PIZZA",

    # Pure Vietnamese Food Classes (Thuần Việt)
    "PHO_BO": "PHO_BO",
    "BEEF_PHO": "PHO_BO",
    "COM_TAM": "COM_TAM",
    "BROKEN_RICE": "COM_TAM",
    "RICE_WHITE_COOKED": "RICE_WHITE_COOKED",
    "PORK_GRILLED_RIB": "PORK_GRILLED_RIB",
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

    # Dataset Label Aliases to Standard Codes
    "STEAMED RICE ROLLS": "BANH_CUON",
    "VIETNAMESE PYRAMID RICE DUMPLING": "BANH_GIO",
    "GRILLED PORK BANH MI": "BANH_MI",
    "HUE-STYLE SPICY BEEF NOODLE SOUP": "BUN_BO_HUE",
    "GRILLED PORK WITH VERMICELLI": "BUN_CHA_HA_NOI",
    "VERMICELLI WITH TOFU": "BUN_DAU_MAM_TOM",
    "GRILLED PORK VERMICELLI": "BUN_THIT_NUONG",
    "PORK OFFAL PORRIDGE": "CHAO_LONG",
    "FLATTENED GREEN RICE": "COM_DEP",
    "BROKEN RICE": "COM_TAM",
    "COM TAM": "COM_TAM",
    "FRIED RICE": "COM_RANG",
    "BEEF PHO": "PHO_BO",
    "PHO BO": "PHO_BO",
    "SPRING ROLLS": "GOI_CUON",
    "STICKY RICE": "XOI",
}


class FoodSegmentationEngine:
    """
    Production YOLOv8 Instance Segmentation inference engine.
    Features single-decode reuse, threshold-gated mask extraction,
    and fast early-exit orientation recovery for rotated mobile frames.
    """

    def __init__(self, weights_path: Optional[str] = None):
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
            self.weights_path = target_weights
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
        scan_min_conf: Optional[float] = None,
        auto_orient: bool = True,
        decoded_img=None,
    ) -> Tuple[List[SegmentedFoodItem], List[SegmentedFoodItem]]:
        """
        Runs YOLOv8-seg inference on image bytes (or a pre-decoded OpenCV BGR ndarray).
        - Uses `scan_min_conf` (defaults to `conf_threshold`) to avoid generating masks for noise.
        - Performs fast orientation recovery (90° CW, 270° CW, 180°) with immediate early-exit
          once qualified items are found.
        """
        if self.model is None:
            return [], []

        try:
            import cv2
            import numpy as np

            if decoded_img is not None:
                img = decoded_img
            else:
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return [], []

            orig_h, orig_w = img.shape[:2]
            effective_scan_conf = conf_threshold if scan_min_conf is None else min(scan_min_conf, conf_threshold)

            # 1. Primary pass at 0° orientation
            results = self.model(img, conf=effective_scan_conf, verbose=False)[0]
            all_detected = self._extract_items_from_results(results, orig_w, orig_h)
            qualified_items = [it for it in all_detected if it.confidence >= conf_threshold]

            # 2. Fast Orientation Recovery with Early Exit (only if 0° found nothing qualified)
            if len(qualified_items) == 0 and auto_orient:
                orientations = (
                    ("rot_90", cv2.ROTATE_90_CLOCKWISE),
                    ("rot_270", cv2.ROTATE_90_COUNTERCLOCKWISE),
                    ("rot_180", cv2.ROTATE_180),
                )

                for name, rot_code in orientations:
                    rot_img = cv2.rotate(img, rot_code)
                    rot_h, rot_w = rot_img.shape[:2]
                    # Filter directly at conf_threshold during recovery to avoid wasting CPU on noise masks
                    rot_res = self.model(rot_img, conf=conf_threshold, verbose=False)[0]
                    rot_raw_items = self._extract_items_from_results(rot_res, rot_w, rot_h)

                    if rot_raw_items:
                        mapped_items = self._remap_items_to_original(rot_raw_items, name, orig_w, orig_h)
                        rot_qual = [it for it in mapped_items if it.confidence >= conf_threshold]
                        if rot_qual:
                            logger.info(
                                f"🔄 Auto-Orientation Applied: Recovered {len(rot_qual)} item(s) via {name} (early exit)"
                            )
                            qualified_items = rot_qual
                            all_detected = mapped_items
                            break

            all_detected.sort(key=lambda x: x.confidence, reverse=True)
            qualified_items.sort(key=lambda x: x.confidence, reverse=True)
            return qualified_items, all_detected

        except Exception as ex:
            logger.warning(f"Inference runtime error: {type(ex).__name__}: {ex}")
            return [], []

    def _extract_items_from_results(self, results, w: int, h: int) -> List[SegmentedFoodItem]:
        items: List[SegmentedFoodItem] = []
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
                        [round(bbox[0] / w, 3), round(bbox[1] / h, 3)],
                        [round(bbox[2] / w, 3), round(bbox[1] / h, 3)],
                        [round(bbox[2] / w, 3), round(bbox[3] / h, 3)],
                        [round(bbox[0] / w, 3), round(bbox[3] / h, 3)],
                        [round(bbox[0] / w, 3), round(bbox[1] / h, 3)],
                    ]

                items.append(SegmentedFoodItem(
                    foodCode=food_code,
                    confidence=confidence,
                    polygonNormalized=polygon,
                    boundingBox=bbox,
                ))
        return items

    def _remap_items_to_original(
        self,
        items: List[SegmentedFoodItem],
        orientation: str,
        orig_w: int,
        orig_h: int,
    ) -> List[SegmentedFoodItem]:
        remapped: List[SegmentedFoodItem] = []
        for it in items:
            new_poly = []
            for p in it.polygonNormalized:
                u_r, v_r = p[0], p[1]
                if orientation == "rot_90":  # Clockwise 90°
                    u, v = v_r, 1.0 - u_r
                elif orientation == "rot_270":  # Counter-Clockwise 90°
                    u, v = 1.0 - v_r, u_r
                elif orientation == "rot_180":
                    u, v = 1.0 - u_r, 1.0 - v_r
                else:
                    u, v = u_r, v_r
                new_poly.append([round(max(0.0, min(1.0, u)), 3), round(max(0.0, min(1.0, v)), 3)])

            all_x = [p[0] * orig_w for p in new_poly]
            all_y = [p[1] * orig_h for p in new_poly]
            new_bbox = [
                round(min(all_x), 2),
                round(min(all_y), 2),
                round(max(all_x), 2),
                round(max(all_y), 2),
            ]

            remapped.append(SegmentedFoodItem(
                foodCode=it.foodCode,
                confidence=it.confidence,
                polygonNormalized=new_poly,
                boundingBox=new_bbox,
            ))
        return remapped
