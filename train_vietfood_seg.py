# ai-service/train_vietfood_seg.py
"""
NutriLens - Class-Balanced YOLOv8 Instance Segmentation Training Pipeline (30 Classes).
Implements `BalancedSegTrainer(SegmentationTrainer)` with:
1. Native Ultralytics `model.class_weights` (smoothed inverse-frequency BCE classification weights).
2. PyTorch `WeightedRandomSampler` so rare classes appear evenly in every training epoch.
3. Cosine LR schedule, Copy-Paste/Mosaic augmentation, `close_mosaic=15`, and optional backbone freeze.
"""
import argparse
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.yolo.segment import SegmentationTrainer


def compute_dataset_class_and_sample_weights(
    dataset,
    num_classes: int,
    cls_beta: float = 0.5,
    sampler_alpha: float = 0.5,
    min_w: float = 0.5,
    max_w: float = 3.0,
) -> tuple[List[float], List[float]]:
    """
    Computes:
    - Per-class loss weights `w_c = clip((median / count_c)^cls_beta, min_w, max_w)` (mean-normalized)
    - Per-image sampling weights `s_i = max((median / count_c)^sampler_alpha for c in image_i)`
    """
    class_counts: Counter = Counter()
    image_classes: List[List[int]] = []

    labels = getattr(dataset, "labels", []) or []
    for lbl in labels:
        cls_arr = lbl.get("cls", [])
        cids = [int(c) for c in np.asarray(cls_arr).reshape(-1).tolist()]
        image_classes.append(cids)
        for cid in cids:
            if 0 <= cid < num_classes:
                class_counts[cid] += 1

    active_counts = [v for v in class_counts.values() if v > 0]
    if not active_counts:
        return [1.0] * num_classes, [1.0] * max(1, len(labels))

    median_cnt = float(sorted(active_counts)[len(active_counts) // 2])

    raw_cls_w = {}
    for cid in range(num_classes):
        cnt = class_counts.get(cid, 0)
        if cnt > 0:
            raw_cls_w[cid] = max(min_w, min(max_w, (median_cnt / float(cnt)) ** cls_beta))

    mean_active = sum(raw_cls_w.values()) / max(1, len(raw_cls_w))
    class_weights = [
        round(raw_cls_w[cid] / mean_active, 4) if cid in raw_cls_w else 1.0
        for cid in range(num_classes)
    ]

    sample_weights: List[float] = []
    for cids in image_classes:
        valid_cids = [c for c in cids if class_counts.get(c, 0) > 0]
        if not valid_cids:
            sample_weights.append(1.0)
        else:
            # Weight image by its rarest class so minority classes are sampled evenly
            w = max((median_cnt / float(class_counts[c])) ** sampler_alpha for c in valid_cids)
            sample_weights.append(float(max(0.5, min(6.0, w))))

    return class_weights, sample_weights


class BalancedSegTrainer(SegmentationTrainer):
    """
    Ultralytics SegmentationTrainer subclass that mitigates severe class imbalance using:
    - WeightedRandomSampler during training epochs
    - Smoothed inverse-frequency `model.class_weights` consumed natively by v8SegmentationLoss
    """

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train" or rank != -1 and rank != 0:
            return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

        dataset = self.build_dataset(dataset_path, mode="train", batch=batch_size)
        num_classes = int(self.data.get("nc", 30))
        class_weights, sample_weights = compute_dataset_class_and_sample_weights(
            dataset,
            num_classes=num_classes,
            cls_beta=0.5,
            sampler_alpha=0.5,
        )

        # Inject native class weights into model so v8SegmentationLoss applies them to BCE cls loss
        if self.model is not None:
            self.model.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            active_summary = ", ".join(
                f"{cid}:{w:.2f}" for cid, w in enumerate(class_weights) if abs(w - 1.0) > 0.05
            )
            print(f"[BalancedSegTrainer] Injected smoothed class_weights ({active_summary[:180]}...)")

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        batch_size = min(batch_size, len(dataset))
        workers = min(os.cpu_count() or 1, batch_size if batch_size > 1 else 0, self.args.workers)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # Mutually exclusive with WeightedRandomSampler
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
        )


def train_segmentation(
    data_yaml: str = "dataset_balanced/data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    base_model: str = "yolov8s-seg.pt",
    batch: int = 16,
    device=None,
    freeze: int = 0,
):
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == 0 else "CPU"
    print("=" * 68)
    print("  AR NutriLens AI - Balanced YOLOv8 Instance Segmentation Trainer")
    print("=" * 68)
    print(f"  Base Weights  : {base_model}")
    print(f"  Dataset YAML  : {data_yaml}")
    print(f"  Epochs / Img  : {epochs} epochs @ {imgsz}x{imgsz} (batch={batch}, freeze={freeze})")
    print(f"  Compute       : {device} ({device_name})")
    print("  Balancing     : WeightedRandomSampler + Class-Balanced BCE Loss + Copy-Paste")
    print("=" * 68 + "\n")

    overrides = {
        "model": base_model,
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": 2 if os.name == "nt" else 4,
        "name": "nutrilens_30classes_balanced_seg",
        "exist_ok": True,
        "amp": True,
        "cos_lr": True,
        "patience": 25,
        "freeze": freeze if freeze > 0 else None,
        # Augmentations tailored for mixed-plate dishes & minority atomic ingredients
        "mosaic": 1.0,
        "close_mosaic": 15,
        "mixup": 0.15,
        "copy_paste": 0.30,
        "hsv_h": 0.015,
        "hsv_s": 0.70,
        "hsv_v": 0.40,
        "degrees": 10.0,
        "fliplr": 0.50,
        "flipud": 0.0,  # Disable vertical upside-down flip for realistic table/plate geometry
        "save": True,
        "plots": True,
    }
    if overrides["freeze"] is None:
        overrides.pop("freeze")

    trainer = BalancedSegTrainer(overrides=overrides)
    trainer.train()

    best_weights = Path(trainer.save_dir) / "weights" / "best.pt"
    target_weights = Path("weights") / "yolov8_vietfood_seg.pt"
    if best_weights.exists():
        target_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, target_weights)
        print(f"[SUCCESS] Deployed balanced weights to: {target_weights}")
    else:
        print(f"[WARNING] best.pt not found at {best_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-Balanced YOLOv8-seg Training for AR-NutriLens")
    parser.add_argument("--data", default="dataset_balanced/data.yaml", help="Path to balanced data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="Base weights (.pt)")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers (e.g. 10 when fine-tuning)")
    parser.add_argument("--device", default=None, help="Device id or 'cpu'")
    args = parser.parse_args()

    train_segmentation(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        base_model=args.model,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
    )
