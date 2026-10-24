"""
NutriLens - YOLOv8 Training & Fine-Tuning Pipeline for Real-World Vietnamese Food
Supports both Detection (yolov8n.pt) and Instance Segmentation (yolov8n-seg.pt).
Usage:
    python train_vietfood.py --data vietfood_seg.yaml --epochs 100 --imgsz 640 --model yolov8n-seg.pt
"""
import os
import argparse
import shutil
from ultralytics import YOLO

def train(data_yaml="vietfood_seg.yaml", epochs=100, imgsz=640, base_model="yolov8n-seg.pt", batch=8, device=None):
    print("=================================================================")
    print("  NutriLens AI - Training YOLOv8 for Vietnamese Street Food")
    print(f"  Base Pretrained Model : {base_model}")
    print(f"  Dataset Config        : {data_yaml}")
    print(f"  Epochs                : {epochs}")
    print(f"  Image Size            : {imgsz}")
    print(f"  Batch Size            : {batch}")
    print(f"  Device                : {device if device is not None else 'auto'}")
    print("=================================================================")

    # Initialize model from official pretrained weights
    model = YOLO(base_model)

    # Train model on dataset
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2 if os.name == "nt" else 4,
        name="nutrilens_vietfood",
        device=device,
        amp=True,
        patience=25,
        exist_ok=True
    )

    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join("runs", "segment", "nutrilens_vietfood")
    best_weights = os.path.join(save_dir, "weights", "best.pt")
    target_weights = os.path.join("weights", "yolov8_vietfood_seg.pt")

    print("\n=================================================================")
    print("  TRAINING COMPLETE!")
    print(f"  Best Weights Path: {best_weights}")
    if os.path.exists(best_weights):
        os.makedirs("weights", exist_ok=True)
        shutil.copy2(best_weights, target_weights)
        print(f"  Successfully copied to: {target_weights}")
    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Vietnamese food dataset")
    parser.add_argument("--data", default="vietfood_seg.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Base model weights (default: yolov8n-seg.pt)")
    parser.add_argument("--device", default=None, help="Device to use: 0, 1, 'cpu' (default: auto)")
    args = parser.parse_args()

    train(data_yaml=args.data, epochs=args.epochs, imgsz=args.imgsz, base_model=args.model, batch=args.batch, device=args.device)
