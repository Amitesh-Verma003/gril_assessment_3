"""
visualize.py – helpers for drawing PPE detection results.
"""

import cv2
import numpy as np
from pathlib import Path

CLASS_NAMES = [
    "person", "hardhat", "hardhat_wrong", "no_hardhat",
    "vest", "vest_wrong", "no_vest", "gloves", "no_gloves",
]

# Green = compliant, Red = violation, Blue = person
COLORS = {
    0: (255, 165,   0),   # person       – orange
    1: (  0, 200,   0),   # hardhat      – green
    2: (  0,   0, 220),   # hardhat_wrong– blue
    3: (  0,   0, 220),   # no_hardhat   – red
    4: (  0, 200,   0),   # vest         – green
    5: (  0,   0, 220),   # vest_wrong   – blue
    6: (220,   0,   0),   # no_vest      – red
    7: (  0, 200,   0),   # gloves       – green
    8: (220,   0,   0),   # no_gloves    – red
}

VIOLATION_IDS = {2, 3, 5, 6, 8}


def draw_boxes(image: np.ndarray, boxes, class_ids, confs) -> np.ndarray:
    """
    Draw bounding boxes on image.
    boxes    : list of [x1,y1,x2,y2] in pixel coords
    class_ids: list of int
    confs    : list of float
    """
    img = image.copy()
    has_violation = False

    for box, cls_id, conf in zip(boxes, class_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if cls_id in VIOLATION_IDS:
            has_violation = True

    # Banner
    banner = "⚠ PPE VIOLATION DETECTED" if has_violation else "✓ PPE COMPLIANT"
    banner_color = (0, 0, 180) if has_violation else (0, 160, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), banner_color, -1)
    cv2.putText(img, banner, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def save_visualization(src_path: str, result, out_dir: str = "outputs/visualizations"):
    """Save a visualized detection result to disk."""
    import cv2 as _cv2
    img = _cv2.imread(str(src_path))
    boxes    = result.boxes.xyxy.cpu().numpy()
    cls_ids  = result.boxes.cls.int().cpu().tolist()
    confs    = result.boxes.conf.cpu().tolist()
    out      = draw_boxes(img, boxes, cls_ids, confs)
    out_path = Path(out_dir) / Path(src_path).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _cv2.imwrite(str(out_path), out)
    return str(out_path)
