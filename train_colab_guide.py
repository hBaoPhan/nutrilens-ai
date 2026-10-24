"""
=============================================================================
  AR NutriLens - High-Performance GPU Training Guide (Google Colab / Cloud GPU)
  Target: YOLOv8 Instance Segmentation (100 Epochs, 14 Vietnamese Food Classes)
=============================================================================

If you wish to train on a high-speed Cloud GPU (Google Colab T4 / A100):

Step 1: Open Google Colab (https://colab.research.google.com/)
        Select Runtime -> Change runtime type -> T4 GPU (Free)

Step 2: Copy and paste the following cells in order:

--- CELL 1: Setup Ultralytics & Verify GPU ---
!nvidia-smi
!pip install -q ultralytics

--- CELL 2: Download Vietnamese Food Dataset ---
!wget -q https://huggingface.co/datasets/HoagMin/Vietnamese_StreetFood_Dataset/resolve/main/Vietnamese_StreetFood_14Class.zip
!unzip -q Vietnamese_StreetFood_14Class.zip -d /content/dataset

--- CELL 3: Convert Bounding Boxes to Segmentation Polygons ---
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
print("Conversion to segmentation polygons complete!")

--- CELL 4: Create Segmentation YAML ---
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

--- CELL 5: Train YOLOv8-seg for 100 Epochs (T4 GPU ~ 15 mins) ---
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
results = model.train(
    data="/content/dataset/data_seg.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    amp=True,
    patience=25,
    name="nutrilens_vietfood_seg_100e"
)

--- CELL 6: Download Best Weights ---
from google.colab import files
files.download(f"{results.save_dir}/weights/best.pt")

Step 3: Save the downloaded `best.pt` file into your local project:
        `ai-service/weights/yolov8_vietfood_seg.pt`
        and restart the `nutrilens_ai` container!
=============================================================================
"""
