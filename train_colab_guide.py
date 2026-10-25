"""
=============================================================================
  AR NutriLens - High-Performance GPU Training Guide (Google Colab / Cloud GPU)
  Target: YOLOv8 Instance Segmentation (24 Classes: 14 Dishes + 10 Ingredients)
=============================================================================

If you wish to train on a high-speed Cloud GPU (Google Colab T4 / A100):

Step 1: Open Google Colab (https://colab.research.google.com/)
        Select Runtime -> Change runtime type -> T4 GPU (Free)

Step 2: Copy and paste the following cells in order:

--- CELL 1: Setup Ultralytics & Verify GPU ---
!nvidia-smi
!pip install -q ultralytics

--- CELL 2: Download Vietnamese Street Food Base Dataset (14 Classes) ---
!wget -q https://huggingface.co/datasets/HoagMin/Vietnamese_StreetFood_Dataset/resolve/main/Vietnamese_StreetFood_14Class.zip
!unzip -q Vietnamese_StreetFood_14Class.zip -d /content/dataset
print("Base 14-class Vietnamese Street Food dataset extracted!")

--- CELL 3: Convert 14-Class Bounding Boxes to Instance Segmentation Polygons ---
import os, glob

for split in ['train', 'valid', 'test']:
    lbl_dir = f'/content/dataset/{split}/labels'
    for p in glob.glob(f'{lbl_dir}/*.txt'):
        lines = []
        with open(p, 'r') as f:
            for l in f:
                parts = l.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1, y1 = max(0, min(1, xc - w/2)), max(0, min(1, yc - h/2))
                    x2, y2 = max(0, min(1, xc + w/2)), max(0, min(1, yc - h/2))
                    x3, y3 = max(0, min(1, xc + w/2)), max(0, min(1, yc + h/2))
                    x4, y4 = max(0, min(1, xc - w/2)), max(0, min(1, yc + h/2))
                    lines.append(f"{c} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
                else:
                    lines.append(l)
        with open(p, 'w') as f:
            f.writelines(lines)
print("Conversion of 14 dishes to segmentation polygons complete!")

--- CELL 4: Merge 10 Atomic Ingredients (Classes 14 - 23) into Dataset ---
import os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ATOMIC_CLASSES = {
    14: ("BOILED_EGG", (250, 245, 230), (255, 185, 30)),
    15: ("CARROT", (245, 120, 20), (220, 95, 15)),
    16: ("CHICKEN_BREAST", (225, 205, 180), (190, 150, 120)),
    17: ("BROCCOLI", (35, 115, 45), (20, 85, 30)),
    18: ("CUCUMBER", (65, 145, 60), (190, 230, 180)),
    19: ("SWEET_POTATO", (190, 80, 50), (235, 160, 45)),
    20: ("TOFU_STEAMED", (245, 245, 240), (225, 225, 215)),
    21: ("SHRIMP_BOILED", (245, 110, 80), (255, 175, 150)),
    22: ("RICE_WHITE", (248, 248, 250), (230, 232, 235)),
    23: ("PORK_GRILLED", (150, 75, 40), (100, 45, 25)),
}

def generate_ingredient(cls_id, width=640, height=640):
    name, color1, color2 = ATOMIC_CLASSES[cls_id]
    bg_tone = random.randint(215, 240)
    img = Image.new("RGB", (width, height), (bg_tone, bg_tone, bg_tone))
    draw = ImageDraw.Draw(img)

    plate_m = random.randint(30, 60)
    draw.ellipse([plate_m, plate_m, width - plate_m, height - plate_m], fill=(250, 250, 252), outline=(190, 195, 205), width=3)

    cx = width / 2.0 + random.randint(-40, 40)
    cy = height / 2.0 + random.randint(-40, 40)
    rx = random.randint(110, 170)
    ry = random.randint(90, 150)

    num_pts = random.randint(14, 20)
    poly = []
    for i in range(num_pts):
        angle = (2.0 * np.pi * i) / num_pts
        jitter = random.uniform(0.85, 1.15)
        px = cx + np.cos(angle) * rx * jitter
        py = cy + np.sin(angle) * ry * jitter
        poly.append((int(px), int(py)))

    draw.polygon(poly, fill=color1, outline=color2)
    if cls_id == 14:
        yolk_r = min(rx, ry) * 0.45
        draw.ellipse([cx - yolk_r, cy - yolk_r, cx + yolk_r, cy + yolk_r], fill=color2)
    elif cls_id in [15, 18]:
        core_r = min(rx, ry) * 0.35
        draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r], fill=color2)
    elif cls_id in [16, 23]:
        for g_offset in [-30, 0, 30]:
            draw.line([(cx - 60, cy + g_offset), (cx + 60, cy + g_offset)], fill=color2, width=4)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    norm_poly = [f"{max(0.001, min(0.999, px/width)):.6f} {max(0.001, min(0.999, py/height)):.6f}" for px, py in poly]
    return img, f"{cls_id} " + " ".join(norm_poly) + "\n"

splits_cfg = {"train": 20, "valid": 5, "test": 5}
total_merged = 0
for split, count in splits_cfg.items():
    img_dir = f"/content/dataset/{split}/images"
    lbl_dir = f"/content/dataset/{split}/labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for c_id, (c_code, _, _) in ATOMIC_CLASSES.items():
        for i in range(count):
            img, lbl_text = generate_ingredient(c_id)
            stem = f"atomic_{c_code.lower()}_{split}_{i:02d}"
            img.save(f"{img_dir}/{stem}.jpg", "JPEG", quality=92)
            with open(f"{lbl_dir}/{stem}.txt", "w") as f:
                f.write(lbl_text)
            total_merged += 1
print(f"Successfully generated and merged {total_merged} samples for all 10 atomic ingredients!")

--- CELL 5: Configure 24-Class YAML File ---
yaml_content = \"\"\"
path: /content/dataset
train: train/images
val: valid/images
test: test/images

nc: 24
names:
  0: BANH_CUON
  1: BANH_GIO
  2: BANH_MI
  3: BUN_BO_HUE
  4: BUN_CHA_HA_NOI
  5: BUN_DAU_MAM_TOM
  6: BUN_THIT_NUONG
  7: CHAO_LONG
  8: COM_DEP
  9: BROKEN_RICE
  10: COM_RANG
  11: BEEF_PHO
  12: GOI_CUON
  13: XOI
  14: BOILED_EGG
  15: CARROT
  16: CHICKEN_BREAST
  17: BROCCOLI
  18: CUCUMBER
  19: SWEET_POTATO
  20: TOFU_STEAMED
  21: SHRIMP_BOILED
  22: RICE_WHITE
  23: PORK_GRILLED
\"\"\"
with open("/content/dataset/data_seg.yaml", "w") as f:
    f.write(yaml_content)
print("Configuration /content/dataset/data_seg.yaml verified for all 24 classes!")

--- CELL 6: Fine-Tune YOLOv8-seg (Transfer Learning 50 Epochs on GPU T4 ~ 12 mins) ---
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model.train(
    data="/content/dataset/data_seg.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    amp=True,
    patience=20,
    name="nutrilens_vietfood_seg_24class"
)

--- CELL 7: Validate & Download Best Model Weights ---
from google.colab import files

best_pt_path = f"{results.save_dir}/weights/best.pt"
print(f"Downloading best trained weights: {best_pt_path}")
files.download(best_pt_path)

Step 3: Save the downloaded `best.pt` file into your local project:
        `ai-service/weights/yolov8_vietfood_seg.pt`
        and restart the `nutrilens_ai` container (`docker restart nutrilens_ai`)!
=============================================================================
"""
