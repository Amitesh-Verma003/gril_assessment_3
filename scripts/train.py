"""
train.py
--------
Train YOLOv8 on the PPE-compliance dataset.

Usage (from project root):
    python scripts/train.py --model yolov8n --epochs 50 --batch 16 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


PRETRAINED_WEIGHTS = {
    "yolov8n": "yolov8n.pt",   # nano   – fastest, least accurate
    "yolov8s": "yolov8s.pt",   # small
    "yolov8m": "yolov8m.pt",   # medium
    "yolov8l": "yolov8l.pt",   # large
    "yolov8x": "yolov8x.pt",   # extra-large – slowest, most accurate
}


def main(args):
    weights = PRETRAINED_WEIGHTS.get(args.model, args.model)
    model   = YOLO(weights)

    data_yaml = Path("dataset/ppe_dataset.yaml").resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    results = model.train(
        data        = str(data_yaml),
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        device      = args.device,          # 0 = first GPU, 'cpu' for CPU
        workers     = args.workers,
        project     = "runs/train",
        name        = f"ppe_{args.model}",
        exist_ok    = True,
        patience    = 20,                   # early-stop after 20 stale epochs
        optimizer   = "AdamW",
        lr0         = 1e-3,
        lrf         = 0.01,
        momentum    = 0.937,
        weight_decay= 5e-4,
        warmup_epochs= 3,
        augment     = True,
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.1,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 10.0,
        translate   = 0.1,
        scale       = 0.5,
        fliplr      = 0.5,
        save_period = 10,                   # save checkpoint every 10 epochs
        val         = True,
        plots       = True,
    )
    print("[DONE] Training complete.")
    print(f"[INFO] Best weights saved to: runs/train/ppe_{args.model}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8n",
                        choices=list(PRETRAINED_WEIGHTS.keys()),
                        help="YOLO variant to use as backbone")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="0",
                        help="GPU index (0,1,…) or 'cpu'")
    parser.add_argument("--workers", type=int,   default=4)
    args = parser.parse_args()
    main(args)
