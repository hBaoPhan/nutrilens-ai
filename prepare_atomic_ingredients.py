"""
AR NutriLens - Atomic Food Ingredients Dataset Generator & Merger (Classes 14 - 23)
Generates structured annotated segmentation data for the 10 atomic ingredients
and merges them into dataset/train, dataset/valid, and dataset/test.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ATOMIC_CLASSES = {
    14: ("BOILED_EGG", (250, 245, 230), (255, 185, 30)),     # White & Yellow Yolk
    15: ("CARROT", (245, 120, 20), (220, 95, 15)),            # Vivid Orange
    16: ("CHICKEN_BREAST", (225, 205, 180), (190, 150, 120)), # Pale cooked poultry
    17: ("BROCCOLI", (35, 115, 45), (20, 85, 30)),            # Deep Green Floret
    18: ("CUCUMBER", (65, 145, 60), (190, 230, 180)),         # Green skin & light core
    19: ("SWEET_POTATO", (190, 80, 50), (235, 160, 45)),      # Purple/Orange flesh
    20: ("TOFU_STEAMED", (245, 245, 240), (225, 225, 215)),   # Soft Cream White
    21: ("SHRIMP_BOILED", (245, 110, 80), (255, 175, 150)),   # Coral Pink & Orange
    22: ("RICE_WHITE", (248, 248, 250), (230, 232, 235)),     # Pearl White grains
    23: ("PORK_GRILLED", (150, 75, 40), (100, 45, 25)),       # Savory Charred Brown
}

def generate_ingredient_sample(cls_id, width=640, height=640):
    name, color1, color2 = ATOMIC_CLASSES[cls_id]
    
    # Background: clean dining surface / plate (light gray / marble / ceramic)
    bg_tone = random.randint(215, 240)
    img = Image.new("RGB", (width, height), (bg_tone, bg_tone, bg_tone))
    draw = ImageDraw.Draw(img)

    # Plate ellipse
    plate_margin = random.randint(30, 60)
    plate_box = [plate_margin, plate_margin, width - plate_margin, height - plate_margin]
    draw.ellipse(plate_box, fill=(250, 250, 252), outline=(190, 195, 205), width=3)

    # Ingredient geometry
    cx = width / 2.0 + random.randint(-40, 40)
    cy = height / 2.0 + random.randint(-40, 40)
    rx = random.randint(110, 170)
    ry = random.randint(90, 150)

    # Generate irregular organic polygon vertices
    num_pts = random.randint(14, 20)
    poly = []
    for i in range(num_pts):
        angle = (2.0 * np.pi * i) / num_pts
        jitter = random.uniform(0.85, 1.15)
        px = cx + np.cos(angle) * rx * jitter
        py = cy + np.sin(angle) * ry * jitter
        poly.append((int(px), int(py)))

    # Draw base ingredient body with soft gradient/shading
    draw.polygon(poly, fill=color1, outline=color2)

    # Inner core detail (e.g. egg yolk, carrot rings, chicken striations)
    if cls_id == 14: # Egg yolk
        yolk_r = min(rx, ry) * 0.45
        draw.ellipse([cx - yolk_r, cy - yolk_r, cx + yolk_r, cy + yolk_r], fill=color2)
    elif cls_id in [15, 18]: # Ring cross section
        core_r = min(rx, ry) * 0.35
        draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=color2)
    elif cls_id in [16, 23]: # Grill marks
        for g_offset in [-30, 0, 30]:
            draw.line([(cx - 60, cy + g_offset), (cx + 60, cy + g_offset)], fill=color2, width=4)

    # Subtle blur to simulate camera depth of field
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Normalized polygon coordinates for YOLO segmentation
    norm_poly = []
    for px, py in poly:
        nx = max(0.001, min(0.999, px / float(width)))
        ny = max(0.001, min(0.999, py / float(height)))
        norm_poly.append(f"{nx:.6f} {ny:.6f}")

    label_str = f"{cls_id} " + " ".join(norm_poly) + "\n"
    return img, label_str

def populate_atomic_ingredients(base_dataset_dir="dataset"):
    splits_config = {
        "train": 15, # 15 images per class for training
        "valid": 4,  # 4 images per class for validation
        "test": 4    # 4 images per class for testing
    }

    total_generated = 0
    for split, count in splits_config.items():
        img_dir = os.path.join(base_dataset_dir, split, "images")
        lbl_dir = os.path.join(base_dataset_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for cls_id in ATOMIC_CLASSES.keys():
            code = ATOMIC_CLASSES[cls_id][0]
            for idx in range(count):
                img, label_content = generate_ingredient_sample(cls_id)
                file_stem = f"atomic_{code.lower()}_{split}_{idx:02d}"
                
                img_path = os.path.join(img_dir, f"{file_stem}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{file_stem}.txt")

                img.save(img_path, "JPEG", quality=92)
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write(label_content)

                total_generated += 1

    print(f"✅ Generated and merged {total_generated} annotated segmentation samples for 10 atomic ingredients across {base_dataset_dir}!")

if __name__ == "__main__":
    target_dir = os.path.join("ai-service", "dataset")
    if not os.path.exists(target_dir):
        target_dir = "dataset"
    populate_atomic_ingredients(target_dir)
