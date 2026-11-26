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
        data         = str(data_yaml),
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        device       = args.device,          # 0 = first GPU, 'cpu' for CPU
        workers      = args.workers,
        project      = "runs/train",
        name         = f"ppe_{args.model}",
        exist_ok     = True,
        patience     = 30,                   # allow longer convergence on small dataset
        optimizer    = "AdamW",
        lr0          = 1e-3,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 5e-4,
        warmup_epochs= 5,
        cls          = 1.0,                  # boost classification weight to separate gear vs hair
        augment      = True,
        mosaic       = 1.0,
        close_mosaic = 15,                   # disable mosaic in final 15 epochs for crisp boxes
        mixup        = 0.15,
        copy_paste   = 0.2,
        hsv_h        = 0.15,                 # wide hue shift so model does not overfit yellow/orange
        hsv_s        = 0.9,                  # strong saturation jitter for white/grey/black helmets
        hsv_v        = 0.6,                  # brightness/lighting robustness
        erasing      = 0.25,
        degrees      = 15.0,
        translate    = 0.1,
        scale        = 0.5,
        fliplr       = 0.5,
        save_period  = 10,                   # save checkpoint every 10 epochs
        val          = True,
        plots        = True,
    )
    best_pt = Path(f"runs/train/ppe_{args.model}/weights/best.pt")
    print("[DONE] Training complete.")
    print(f"[INFO] Best weights saved to: {best_pt}")

    if args.export_onnx and best_pt.exists():
        best_model = YOLO(str(best_pt))
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"[INFO] Exported ONNX model to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8s",
                        choices=list(PRETRAINED_WEIGHTS.keys()),
                        help="YOLO variant to use as backbone")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="0",
                        help="GPU index (0,1,…) or 'cpu'")
    parser.add_argument("--workers", type=int,   default=4)
    parser.add_argument("--export_onnx", action="store_true",
                        help="Also export best.pt to ONNX after training")
    args = parser.parse_args()
    main(args)
