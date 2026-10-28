"""
NutriLens - YOLOv8 Instance Segmentation Training Pipeline for Vietnamese Street Food
Trains yolov8n-seg.pt on 14 Vietnamese food classes for 100 epochs at 640x640 resolution.
"""
import os
import argparse
import shutil
import torch
from ultralytics import YOLO

def train_segmentation(
    data_yaml="dataset/data.yaml",
    epochs=80,
    imgsz=640,
    base_model="yolov8s-seg.pt",
    batch=8,
    device=None
):
    # Auto-detect optimal device (NVIDIA GPU if available)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device == 0 else "CPU"

    print("=================================================================")
    print("  AR NutriLens AI - Training YOLOv8 Instance Segmentation")
    print("=================================================================")
    print(f"  Target Architecture   : {base_model} (Instance Segmentation)")
    print(f"  Dataset Configuration : {data_yaml}")
    print(f"  Total Epochs          : {epochs}")
    print(f"  Input Resolution      : {imgsz}x{imgsz}")
    print(f"  Batch Size            : {batch}")
    print(f"  Compute Device        : {device} ({device_name})")
    print("=================================================================\n")

    # 1. Initialize YOLOv8 segmentation model
    model = YOLO(base_model)

    # 2. Run high-performance training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2 if os.name == "nt" else 4,
        name="nutrilens_vietfood_seg",
        exist_ok=True,
        amp=True,              # Automatic Mixed Precision for RTX / Tensor Cores
        patience=25,           # Early stopping patience if validation plateaus
        save=True,
        plots=True
    )

    # 3. Locate best model weights dynamically
    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join("runs", "segment", "nutrilens_vietfood_seg")
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    target_weights = os.path.join("weights", "yolov8_vietfood_seg.pt")

    print("\n=================================================================")
    print("  TRAINING COMPLETE!")
    print(f"  Best Weights Path: {best_weights}")
    
    if os.path.exists(best_weights):
        os.makedirs("weights", exist_ok=True)
        shutil.copy2(best_weights, target_weights)
        print(f"  Successfully deployed best model to: {target_weights}")
    else:
        print(f"  Warning: best.pt was not found at {best_weights}")
    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Instance Segmentation on Vietnamese food")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to segmentation data yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8 for 4GB VRAM)")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="Pretrained base model (default: yolov8s-seg.pt)")
    parser.add_argument("--device", default=None, help="Device to use: 0, 1, 'cpu' (default: auto)")
    
    args = parser.parse_args()
    train_segmentation(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        base_model=args.model,
        batch=args.batch,
        device=args.device
    )
