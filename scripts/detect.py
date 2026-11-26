"""
detect.py
---------
Run inference with a trained PPE-compliance model on images / video / webcam.

Usage:
    python scripts/detect.py --weights runs/train/ppe_yolov8n/weights/best.pt \
                             --source  dataset/images/test \
                             --conf    0.4
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


VIOLATION_CLASSES = {
    2: "hardhat_wrong",
    3: "no_hardhat",
    5: "vest_wrong",
    6: "no_vest",
    8: "no_gloves",
}


def main(args):
    model  = YOLO(args.weights)
    source = args.source

    results = model.predict(
        source      = source,
        conf        = args.conf,
        iou         = args.iou,
        imgsz       = args.imgsz,
        device      = args.device,
        save        = True,
        save_txt    = True,
        save_conf   = True,
        project     = "outputs/predictions",
        name        = "run",
        exist_ok    = True,
    )

    # --- Summarise violations per image ---
    print(f"\n{'='*60}")
    print("PPE VIOLATION SUMMARY")
    print(f"{'='*60}")
    total_violations = 0
    for r in results:
        image_name = Path(r.path).name
        violations = []
        for cls_id in r.boxes.cls.int().tolist():
            if cls_id in VIOLATION_CLASSES:
                violations.append(VIOLATION_CLASSES[cls_id])
        if violations:
            print(f"[VIOLATION] {image_name}: {', '.join(set(violations))}")
            total_violations += len(violations)
        else:
            print(f"[OK]        {image_name}: All PPE compliant")
    print(f"\nTotal violations detected: {total_violations}")
    print(f"Results saved to: outputs/predictions/run/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/train/ppe_yolov8n/weights/best.pt")
    parser.add_argument("--source",  default="dataset/images/test",
                        help="Image dir, video path, or 0 for webcam")
    parser.add_argument("--conf",    type=float, default=0.4)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="0")
    args = parser.parse_args()
    main(args)
