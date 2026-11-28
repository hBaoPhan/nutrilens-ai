# ai-service/rebalance_dataset.py
"""
AR-NutriLens Dataset Rebalancer for YOLOv8 Instance Segmentation (30 Classes).
Solves severe class imbalance at the data level WITHOUT synthetic cartoon drawings:
1. Pools all real photographic samples and performs Stratified Split (80% train / 15% valid / 5% test).
2. Caps over-represented classes in `train` (under-sampling) while protecting multi-label rare images.
3. Applies Repeat Factor Sampling (RFS) + natural photographic augmentations (flip, mild rotation,
   brightness/contrast/HSV jitter) ONLY on `train` to prevent validation data leakage.
4. Exports smoothed inverse-frequency `class_weights.json` for BalancedSegTrainer.
"""
import argparse
import json
import math
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

NUTRILENS_CLASSES: Dict[int, str] = {
    0: "BANH_CUON",
    1: "BANH_GIO",
    2: "BANH_MI",
    3: "BUN_BO_HUE",
    4: "BUN_CHA_HA_NOI",
    5: "BUN_DAU_MAM_TOM",
    6: "BUN_THIT_NUONG",
    7: "CHAO_LONG",
    8: "COM_DEP",
    9: "COM_TAM",
    10: "COM_RANG",
    11: "PHO_BO",
    12: "GOI_CUON",
    13: "XOI",
    14: "BOILED_EGG",
    15: "CARROT",
    16: "CHICKEN_BREAST",
    17: "BROCCOLI",
    18: "CUCUMBER",
    19: "SWEET_POTATO",
    20: "TOFU_STEAMED",
    21: "SHRIMP_BOILED",
    22: "RICE_WHITE",
    23: "PORK_GRILLED",
    24: "ORANGE",
    25: "BANANA",
    26: "APPLE",
    27: "COCONUT",
    28: "SUA_CHUA",
    29: "WATERMELON",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_yolo_seg_label(label_path: Path) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """Parses a YOLOv8 segmentation label file into [(class_id, [(x, y), ...]), ...]."""
    annotations: List[Tuple[int, List[Tuple[float, float]]]] = []
    if not label_path.is_file():
        return annotations
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) < 7:
                continue
            try:
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                pts = [(max(0.0, min(1.0, coords[i])), max(0.0, min(1.0, coords[i + 1]))) for i in range(0, len(coords), 2)]
                if len(pts) >= 3:
                    annotations.append((cid, pts))
            except ValueError:
                continue
    return annotations


def write_yolo_seg_label(label_path: Path, annotations: List[Tuple[int, List[Tuple[float, float]]]]):
    lines = []
    for cid, pts in annotations:
        if len(pts) < 3:
            continue
        flat = " ".join(f"{max(0.0, min(1.0, x)):.6f} {max(0.0, min(1.0, y)):.6f}" for x, y in pts)
        lines.append(f"{cid} {flat}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def collect_samples(dataset_dir: Path) -> List[Dict]:
    """Collects all valid real image-label pairs across train/valid/test splits (skips legacy synth_atomic)."""
    samples: List[Dict] = []
    seen_stems = set()
    for split in ("train", "valid", "val", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if "synth_atomic" in img_path.name or img_path.name.startswith("atomic_"):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            anns = parse_yolo_seg_label(lbl_path)
            if not anns:
                continue
            unique_key = f"{split}_{img_path.stem}"
            if unique_key in seen_stems:
                continue
            seen_stems.add(unique_key)
            classes = sorted({cid for cid, _ in anns})
            samples.append({
                "img_path": img_path,
                "lbl_path": lbl_path,
                "anns": anns,
                "classes": classes,
                "stem": f"{split}_{img_path.stem}",
            })
    return samples


def stratified_split(
    samples: List[Dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split grouped by each image's rarest class so rare classes never lose >60%
    of their data to val/test and every active class gets guaranteed representation in train & valid.
    """
    rng = random.Random(seed)
    global_counts = Counter(cid for s in samples for cid in s["classes"])

    buckets: Dict[int, List[Dict]] = defaultdict(list)
    for s in samples:
        rarest_cid = min(s["classes"], key=lambda c: global_counts[c])
        buckets[rarest_cid].append(s)

    train_set, val_set, test_set = [], [], []
    for cid, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        if n <= 2:
            train_set.extend(items)
            continue
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * max(0.0, 1.0 - train_ratio - val_ratio)))) if n >= 10 else 0
        n_train = max(1, n - n_val - n_test)
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

    return train_set, val_set, test_set


def cap_dominant_classes(
    train_samples: List[Dict],
    max_per_class: int = 250,
    seed: int = 42,
) -> List[Dict]:
    """
    Caps dominant classes in train_samples without dropping images that also contain rare classes.
    """
    rng = random.Random(seed)
    shuffled = list(train_samples)
    rng.shuffle(shuffled)

    raw_counts = Counter(cid for s in shuffled for cid in s["classes"])
    rare_classes = {c for c, cnt in raw_counts.items() if cnt < max_per_class}

    kept: List[Dict] = []
    running_counts: Counter = Counter()

    # Pass 1: Keep all images containing at least one rare class
    remaining = []
    for s in shuffled:
        if any(c in rare_classes for c in s["classes"]):
            kept.append(s)
            for c in s["classes"]:
                running_counts[c] += 1
        else:
            remaining.append(s)

    # Pass 2: Fill dominant-only images up to max_per_class
    for s in remaining:
        if any(running_counts[c] < max_per_class for c in s["classes"]):
            kept.append(s)
            for c in s["classes"]:
                running_counts[c] += 1

    return kept


def augment_image_and_polygons(
    img: np.ndarray,
    anns: List[Tuple[int, List[Tuple[float, float]]]],
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, List[Tuple[float, float]]]]]:
    """
    Natural photographic augmentation for real food images + exact polygon coordinate transform:
    - Random horizontal flip (p=0.5)
    - Mild affine rotation (-15° to +15°) & scale (0.90 to 1.08)
    - Subtle HSV / brightness & contrast lighting jitter
    """
    h, w = img.shape[:2]
    aug_img = img.copy()
    aug_anns = [(cid, list(pts)) for cid, pts in anns]

    # 1. Horizontal flip
    if rng.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        aug_anns = [(cid, [(1.0 - x, y) for x, y in pts]) for cid, pts in aug_anns]

    # 2. Mild rotation & scale around center
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    transformed_anns = []
    for cid, pts in aug_anns:
        new_pts = []
        for x, y in pts:
            px, py = x * w, y * h
            nx = (M[0, 0] * px + M[0, 1] * py + M[0, 2]) / w
            ny = (M[1, 0] * px + M[1, 1] * py + M[1, 2]) / h
            new_pts.append((max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))))
        transformed_anns.append((cid, new_pts))

    # 3. Natural lighting (brightness / contrast / saturation jitter)
    alpha = rng.uniform(0.85, 1.18)  # contrast
    beta = rng.uniform(-18.0, 18.0)  # brightness
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    if rng.random() < 0.5:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.82, 1.20), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return aug_img, transformed_anns


def compute_class_weights(
    counts: Counter,
    num_classes: int = 30,
    beta: float = 0.5,
    min_w: float = 0.5,
    max_w: float = 3.0,
) -> List[float]:
    """
    Computes smoothed inverse-frequency class weights for Ultralytics v8SegmentationLoss:
        w_c = clip((median(N) / N_c) ** beta, min_w, max_w)
    Normalized to mean 1.0 across active classes; inactive classes default to 1.0.
    """
    active_vals = [cnt for cnt in counts.values() if cnt > 0]
    if not active_vals:
        return [1.0] * num_classes

    sorted_vals = sorted(active_vals)
    median_cnt = float(sorted_vals[len(sorted_vals) // 2])

    raw_weights = {}
    for cid in range(num_classes):
        cnt = counts.get(cid, 0)
        if cnt > 0:
            w = (median_cnt / float(cnt)) ** beta
            raw_weights[cid] = max(min_w, min(max_w, w))

    mean_active = sum(raw_weights.values()) / max(1, len(raw_weights))
    final_weights = []
    for cid in range(num_classes):
        if cid in raw_weights:
            final_weights.append(round(raw_weights[cid] / mean_active, 4))
        else:
            final_weights.append(1.0)
    return final_weights


def write_split(samples: List[Dict], out_split_dir: Path):
    img_out = out_split_dir / "images"
    lbl_out = out_split_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        ext = s["img_path"].suffix.lower()
        shutil.copy2(s["img_path"], img_out / f"{s['stem']}{ext}")
        write_yolo_seg_label(lbl_out / f"{s['stem']}.txt", s["anns"])


def apply_rfs_augmentation(
    train_samples: List[Dict],
    out_train_dir: Path,
    target_min: int = 150,
    max_copies_per_image: int = 5,
    seed: int = 42,
) -> Counter:
    """
    Applies Repeat Factor Sampling (RFS) + natural photographic augmentation ONLY to `train`
    for under-represented classes (< target_min), capped at `max_copies_per_image` to avoid overfitting.
    """
    rng = random.Random(seed)
    img_out = out_train_dir / "images"
    lbl_out = out_train_dir / "labels"

    counts = Counter(cid for s in train_samples for cid, _ in s["anns"])
    by_class: Dict[int, List[Dict]] = defaultdict(list)
    for s in train_samples:
        for cid in s["classes"]:
            by_class[cid].append(s)

    aug_created = 0
    for cid, cls_samples in sorted(by_class.items()):
        cur = counts[cid]
        if cur == 0 or cur >= target_min:
            continue
        needed = min(target_min - cur, len(cls_samples) * max_copies_per_image)
        if needed <= 0:
            continue

        for k in range(needed):
            src = cls_samples[k % len(cls_samples)]
            img = cv2.imread(str(src["img_path"]))
            if img is None:
                continue
            aug_img, aug_anns = augment_image_and_polygons(img, src["anns"], rng)
            aug_stem = f"rfs_c{cid}_{k:04d}_{src['stem']}"
            cv2.imwrite(str(img_out / f"{aug_stem}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_yolo_seg_label(lbl_out / f"{aug_stem}.txt", aug_anns)
            for ann_cid, _ in aug_anns:
                counts[ann_cid] += 1
            aug_created += 1

    print(f"[RFS] Generated {aug_created} natural augmented training images for under-represented classes.")
    return counts


def rebalance_dataset(
    src_dir: Path,
    out_dir: Path,
    target_min: int = 150,
    max_per_class: int = 250,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict:
    samples = collect_samples(src_dir)
    if not samples:
        raise RuntimeError(f"No valid YOLO segmentation samples found in {src_dir}")

    before_counts = Counter(cid for s in samples for cid, _ in s["anns"])
    train_s, val_s, test_s = stratified_split(samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    train_capped = cap_dominant_classes(train_s, max_per_class=max_per_class, seed=seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    write_split(train_capped, out_dir / "train")
    write_split(val_s, out_dir / "valid")
    write_split(test_s, out_dir / "test")

    after_train_counts = apply_rfs_augmentation(
        train_capped,
        out_dir / "train",
        target_min=target_min,
        seed=seed,
    )
    val_counts = Counter(cid for s in val_s for cid, _ in s["anns"])
    test_counts = Counter(cid for s in test_s for cid, _ in s["anns"])

    # Write standardized data.yaml
    yaml_lines = [
        f"path: {out_dir.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(NUTRILENS_CLASSES)}",
        "names:",
    ]
    for cid in range(len(NUTRILENS_CLASSES)):
        yaml_lines.append(f"  {cid}: {NUTRILENS_CLASSES[cid]}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # Compute & save class_weights.json for BalancedSegTrainer
    class_weights = compute_class_weights(after_train_counts, num_classes=len(NUTRILENS_CLASSES))
    weights_payload = {
        "num_classes": len(NUTRILENS_CLASSES),
        "train_instance_counts": {str(k): int(v) for k, v in sorted(after_train_counts.items())},
        "class_weights": class_weights,
    }
    (out_dir / "class_weights.json").write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    # Print summary table
    active_before = [v for v in before_counts.values() if v > 0]
    active_after = [v for v in after_train_counts.values() if v > 0]
    ratio_before = (max(active_before) / min(active_before)) if active_before else 0.0
    ratio_after = (max(active_after) / min(active_after)) if active_after else 0.0

    print("\n" + "=" * 86)
    print(f"REBALANCE SUMMARY | Imbalance Ratio: {ratio_before:.1f}:1  --->  {ratio_after:.2f}:1")
    print("=" * 86)
    print(f"{'ID':>2} | {'Class Name':<18} | {'Before Total':>12} | {'Train After':>11} | {'Valid':>5} | {'Test':>4} | {'Weight':>6}")
    print("-" * 86)
    for cid in range(len(NUTRILENS_CLASSES)):
        b = before_counts[cid]
        tr = after_train_counts[cid]
        va = val_counts[cid]
        te = test_counts[cid]
        if b > 0 or tr > 0:
            print(f"{cid:2d} | {NUTRILENS_CLASSES[cid]:<18} | {b:12d} | {tr:11d} | {va:5d} | {te:4d} | {class_weights[cid]:6.2f}")
    print("=" * 86)

    return {
        "ratio_before": round(ratio_before, 2),
        "ratio_after": round(ratio_after, 2),
        "train_counts": dict(after_train_counts),
        "val_counts": dict(val_counts),
        "test_counts": dict(test_counts),
        "class_weights": class_weights,
    }


def main():
    parser = argparse.ArgumentParser(description="Stratified 80/15/5 + Capping + RFS dataset rebalancer")
    parser.add_argument("--src", default="dataset", help="Source YOLO dataset folder (or path to .zip)")
    parser.add_argument("--out", default="dataset_balanced", help="Output balanced dataset folder")
    parser.add_argument("--target-min", type=int, default=150, help="Target minimum train instances per active class")
    parser.add_argument("--max-per-class", type=int, default=250, help="Maximum train images per dominant class")
    parser.add_argument("--zip-out", default="", help="Optional output .zip archive path")
    args = parser.parse_args()

    src_path = Path(args.src)
    tmp_unzip = None
    if src_path.suffix.lower() == ".zip" and src_path.is_file():
        tmp_unzip = src_path.parent / "_tmp_rebalance_src"
        if tmp_unzip.exists():
            shutil.rmtree(tmp_unzip)
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(tmp_unzip)
        src_path = tmp_unzip

    try:
        rebalance_dataset(
            src_dir=src_path,
            out_dir=Path(args.out),
            target_min=args.target_min,
            max_per_class=args.max_per_class,
        )
        if args.zip_out:
            out_zip = Path(args.zip_out)
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(args.out):
                    for fn in files:
                        full = Path(root) / fn
                        zf.write(full, arcname=full.relative_to(args.out))
            print(f"[ZIP] Wrote balanced archive to {out_zip} ({out_zip.stat().st_size / (1024 * 1024):.1f} MB)")
    finally:
        if tmp_unzip and tmp_unzip.exists():
            shutil.rmtree(tmp_unzip)


if __name__ == "__main__":
    main()
