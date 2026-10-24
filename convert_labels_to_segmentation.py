"""
NutriLens - Label Format Converter: Bounding Box to Instance Segmentation Polygon
Converts standard 5-token YOLO bounding boxes [cls, xc, yc, w, h] to 
normalized 4-corner polygons [cls, x1, y1, x2, y2, x3, y3, x4, y4].
Existing polygons (>5 tokens) are strictly preserved.
"""
import os
import glob
import shutil

def convert_labels(dataset_dir="dataset"):
    splits = ["train", "valid", "test"]
    total_converted = 0
    total_preserved = 0
    total_empty = 0

    for split in splits:
        labels_dir = os.path.join(dataset_dir, split, "labels")
        backup_dir = os.path.join(dataset_dir, split, "labels_bbox_backup")

        if not os.path.exists(labels_dir):
            print(f"Directory {labels_dir} does not exist, skipping.")
            continue

        # Create backup if not already backed up
        if not os.path.exists(backup_dir):
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(labels_dir, backup_dir)
        else:
            print(f"Backup already exists at: {backup_dir}")

        txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        split_converted = 0
        split_preserved = 0

        for txt_path in txt_files:
            new_lines = []
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                total_empty += 1
                continue

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                cls_id = parts[0]

                # If standard bounding box: cls xc yc w h
                if len(parts) == 5:
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0.0, min(1.0, xc - w / 2.0))
                    y1 = max(0.0, min(1.0, yc - h / 2.0))
                    x2 = max(0.0, min(1.0, xc + w / 2.0))
                    y2 = max(0.0, min(1.0, yc - h / 2.0))
                    x3 = max(0.0, min(1.0, xc + w / 2.0))
                    y3 = max(0.0, min(1.0, yc + h / 2.0))
                    x4 = max(0.0, min(1.0, xc - w / 2.0))
                    y4 = max(0.0, min(1.0, yc + h / 2.0))

                    # 4 vertices polygon
                    new_line = f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                    new_lines.append(new_line)
                    split_converted += 1
                else:
                    # Already polygon or custom format (> 5 tokens)
                    new_lines.append(line)
                    split_preserved += 1

            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        total_converted += split_converted
        total_preserved += split_preserved
        print(f"[{split}] Converted {split_converted} bounding boxes to polygons | Preserved {split_preserved} polygons")

    print(f"\n=======================================================")
    print(f"Conversion Summary:")
    print(f"  Total bbox labels converted to polygons : {total_converted}")
    print(f"  Total existing polygons preserved       : {total_preserved}")
    print(f"  Total empty label files                 : {total_empty}")
    print(f"=======================================================")

if __name__ == "__main__":
    convert_labels()
