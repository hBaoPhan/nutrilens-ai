"""
NutriLens - Dataset Merger & Class Remapper Tool
Merges additional ingredient datasets (e.g. Boiled Eggs, Vegetables) into the main dataset
and remaps class IDs to match the 24-class hierarchical taxonomy.
"""
import os
import glob
import shutil

TARGET_CLASS_MAP = {
    "egg": 14,
    "boiled_egg": 14,
    "boiled egg": 14,
    "boiled-egg": 14,
    "boiled-eggs": 14,
    "carrot": 15,
    "carrots": 15,
    "chicken_breast": 16,
    "chicken breast": 16,
    "broccoli": 17,
    "cucumber": 18,
    "sweet_potato": 19,
    "sweet potato": 19,
    "tofu": 20,
    "tofu_steamed": 20,
    "shrimp": 21,
    "shrimp_boiled": 21,
    "rice_white": 22,
    "white_rice": 22,
    "pork_grilled": 23
}

def merge_dataset(source_dir, main_dataset_dir="dataset", source_class_mapping=None):
    """
    source_dir: Directory containing train/valid/test splits of the new dataset
    source_class_mapping: dict mapping source class IDs (int) to target class ID (int)
                          e.g. {0: 14} for eggs
    """
    if source_class_mapping is None:
        source_class_mapping = {0: 14} # Default single-class to BOILED_EGG

    splits = ["train", "valid", "test"]
    merged_images = 0
    merged_labels = 0

    for split in splits:
        src_img_dir = os.path.join(source_dir, split, "images")
        src_lbl_dir = os.path.join(source_dir, split, "labels")

        dst_img_dir = os.path.join(main_dataset_dir, split, "images")
        dst_lbl_dir = os.path.join(main_dataset_dir, split, "labels")

        if not os.path.exists(src_img_dir):
            continue

        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for img_path in glob.glob(os.path.join(src_img_dir, "*.*")):
            base_name = os.path.basename(img_path)
            stem, ext = os.path.splitext(base_name)
            if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            lbl_name = f"{stem}.txt"
            src_lbl_path = os.path.join(src_lbl_dir, lbl_name)

            if not os.path.exists(src_lbl_path):
                continue

            # Prefix file name to avoid collision
            prefix = "ingred_"
            dst_img_path = os.path.join(dst_img_dir, f"{prefix}{base_name}")
            dst_lbl_path = os.path.join(dst_lbl_dir, f"{prefix}{lbl_name}")

            # Read and remap label lines
            new_lines = []
            with open(src_lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    src_cls = int(parts[0])
                    if src_cls in source_class_mapping:
                        target_cls = source_class_mapping[src_cls]
                        coords = parts[1:]

                        # If 4 coords (bbox), convert to 4-vertex polygon
                        if len(coords) == 4:
                            xc, yc, w, h = map(float, coords)
                            x1 = max(0.0, min(1.0, xc - w / 2.0))
                            y1 = max(0.0, min(1.0, yc - h / 2.0))
                            x2 = max(0.0, min(1.0, xc + w / 2.0))
                            y2 = max(0.0, min(1.0, yc - h / 2.0))
                            x3 = max(0.0, min(1.0, xc + w / 2.0))
                            y3 = max(0.0, min(1.0, yc + h / 2.0))
                            x4 = max(0.0, min(1.0, xc - w / 2.0))
                            y4 = max(0.0, min(1.0, yc + h / 2.0))
                            new_lines.append(f"{target_cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n")
                        else:
                            # Preserve polygon coords
                            new_lines.append(f"{target_cls} {' '.join(coords)}\n")

            if new_lines:
                shutil.copy2(img_path, dst_img_path)
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                merged_images += 1
                merged_labels += 1

    print(f"Successfully merged {merged_images} images and {merged_labels} labels into {main_dataset_dir}!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        merge_dataset(sys.argv[1])
    else:
        print("Usage: python merge_ingredients_dataset.py <path_to_ingredient_dataset>")
