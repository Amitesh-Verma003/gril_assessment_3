"""
evaluate.py
-----------
Evaluate a trained model on the test split and print mAP metrics.

Usage:
    python scripts/evaluate.py --weights runs/train/ppe_yolov8n/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main(args):
    model     = YOLO(args.weights)
    data_yaml = Path("dataset/ppe_dataset.yaml").resolve()

    metrics = model.val(
        data     = str(data_yaml),
        split    = "test",
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        project  = "runs/detect",
        name     = "eval",
        exist_ok = True,
        plots    = True,
        save_json= True,
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")

    print("\nPer-class AP50:")
    names = model.names
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {names[i]:15s}: {ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/ppe_yolov8n/weights/best.pt")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="0")
    args = parser.parse_args()
    main(args)
