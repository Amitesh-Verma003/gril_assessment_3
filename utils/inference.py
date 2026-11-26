"""
inference.py
------------
Shared helpers for PPE compliance model loading, inference, annotation,
video processing, and dataset/training inspection used by the Streamlit UI.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from utils.visualize import CLASS_NAMES, VIOLATION_IDS

COMPLIANT_IDS = {1, 4, 7}
ANCHOR_IDS = {0}
MISUSE_VIOLATION_IDS = {2, 5}
MISSING_VIOLATION_IDS = {3, 6, 8}

# High-contrast RGB colors for UI badges and bounding boxes
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (56, 189, 248),   # person         - sky blue
    1: (34, 197, 94),    # hardhat        - green
    2: (245, 158, 11),   # hardhat_wrong  - amber
    3: (239, 68, 68),    # no_hardhat     - red
    4: (34, 197, 94),    # vest           - green
    5: (245, 158, 11),   # vest_wrong     - amber
    6: (239, 68, 68),    # no_vest        - red
    7: (34, 197, 94),    # gloves         - green
    8: (239, 68, 68),    # no_gloves      - red
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Detection:
    """Single bounding-box prediction."""

    xyxy: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    category: str  # "Compliant" | "Violation" | "Anchor"


def is_ultralytics_available() -> bool:
    """Return True if ultralytics can be imported in the current environment."""
    return importlib.util.find_spec("ultralytics") is not None


def classify_id(class_id: int) -> str:
    """Map a class index to its compliance category."""
    if class_id in VIOLATION_IDS:
        return "Violation"
    if class_id in COMPLIANT_IDS:
        return "Compliant"
    return "Anchor"


def discover_weights(root: Path = PROJECT_ROOT) -> list[Path]:
    """Find available .pt model checkpoints in models/weights, runs/, and project root."""
    candidates: list[Path] = []
    search_dirs = [root / "models" / "weights", root / "runs", root]
    seen: set[Path] = set()
    for base in search_dirs:
        if not base.exists():
            continue
        pattern = "*.pt" if base == root else "**/*.pt"
        for pt_path in sorted(base.glob(pattern)):
            resolved = pt_path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(pt_path)
    return candidates


def discover_sample_images(root: Path = PROJECT_ROOT, limit: int = 30) -> list[Path]:
    """Find sample images in dataset/images/test, val, or outputs if present."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    search_dirs = [
        root / "dataset" / "images" / "test",
        root / "dataset" / "images" / "val",
        root / "outputs",
    ]
    found: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for p in sorted(directory.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p)
                if len(found) >= limit:
                    return found
    return found


def load_dataset_config(root: Path = PROJECT_ROOT) -> dict:
    """Load dataset/ppe_dataset.yaml if present."""
    cfg_path = root / "dataset" / "ppe_dataset.yaml"
    if not cfg_path.exists():
        return {"nc": len(CLASS_NAMES), "names": dict(enumerate(CLASS_NAMES))}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yolo_model(weights_source: str):
    """Load an Ultralytics YOLO model from a local path or model name."""
    if not is_ultralytics_available():
        raise RuntimeError(
            "The 'ultralytics' package is not installed in this Python environment. "
            "Install it with `pip install ultralytics` or enable Demo Mode in the sidebar."
        )
    from ultralytics import YOLO

    return YOLO(weights_source)


def run_image_inference(
    model,
    image_rgb: np.ndarray,
    conf: float = 0.40,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
) -> list[Detection]:
    """Run YOLOv8 prediction on an RGB numpy array and return Detection objects."""
    # Ultralytics accepts RGB PIL images directly without BGR channel confusion
    pil_img = Image.fromarray(image_rgb)
    predict_kwargs = {
        "source": pil_img,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device and device != "auto":
        predict_kwargs["device"] = device

    results = model.predict(**predict_kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    model_names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.int().cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()

    detections: list[Detection] = []
    for box, cls_id, score in zip(boxes, cls_ids, confs):
        name = (
            model_names.get(cls_id)
            if isinstance(model_names, dict)
            else (CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
        )
        # Resolve category by name when custom/COCO weights are used
        resolved_id = CLASS_NAMES.index(name) if name in CLASS_NAMES else cls_id
        detections.append(
            Detection(
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                class_id=resolved_id,
                class_name=str(name),
                confidence=float(score),
                category=classify_id(resolved_id) if name in CLASS_NAMES else "Anchor",
            )
        )
    return detections


def generate_demo_detections(image_rgb: np.ndarray, conf_threshold: float = 0.40) -> list[Detection]:
    """Generate deterministic synthetic PPE detections for UI preview without weights."""
    h, w = image_rgb.shape[:2]
    candidates = [
        Detection((0.12 * w, 0.16 * h, 0.44 * w, 0.90 * h), 0, "person", 0.94, "Anchor"),
        Detection((0.20 * w, 0.17 * h, 0.35 * w, 0.31 * h), 1, "hardhat", 0.89, "Compliant"),
        Detection((0.17 * w, 0.34 * h, 0.40 * w, 0.64 * h), 4, "vest", 0.86, "Compliant"),
        Detection((0.14 * w, 0.58 * h, 0.22 * w, 0.68 * h), 7, "gloves", 0.78, "Compliant"),
        Detection((0.56 * w, 0.18 * h, 0.86 * w, 0.91 * h), 0, "person", 0.92, "Anchor"),
        Detection((0.63 * w, 0.19 * h, 0.78 * w, 0.33 * h), 3, "no_hardhat", 0.87, "Violation"),
        Detection((0.60 * w, 0.35 * h, 0.82 * w, 0.66 * h), 5, "vest_wrong", 0.74, "Violation"),
        Detection((0.55 * w, 0.60 * h, 0.63 * w, 0.70 * h), 8, "no_gloves", 0.69, "Violation"),
    ]
    return [d for d in candidates if d.confidence >= conf_threshold]


def draw_annotations(
    image_rgb: np.ndarray,
    detections: list[Detection],
    allowed_classes: set[str] | None = None,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 3,
    show_banner: bool = True,
) -> np.ndarray:
    """Draw crisp RGB bounding boxes and a compliance status banner onto an image."""
    pil_img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    filtered = [
        d for d in detections if allowed_classes is None or d.class_name in allowed_classes
    ]
    has_violation = any(d.category == "Violation" for d in detections)

    for det in filtered:
        x1, y1, x2, y2 = det.xyxy
        color = CLASS_COLORS_RGB.get(det.class_id, (148, 163, 184))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if show_labels:
            label = f"{det.class_name} {det.confidence:.2f}" if show_conf else det.class_name
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 4
            label_top = max(0, y1 - th - pad * 2)
            draw.rectangle(
                [x1, label_top, x1 + tw + pad * 2, label_top + th + pad * 2],
                fill=color,
            )
            text_color = (15, 23, 42) if det.class_id in (0, 2, 5) else (255, 255, 255)
            draw.text((x1 + pad, label_top + pad), label, fill=text_color, font=font)

    if show_banner:
        banner_h = 34
        banner_color = (185, 28, 28) if has_violation else (21, 128, 61)
        banner_text = (
            "PPE VIOLATION DETECTED" if has_violation else "PPE COMPLIANT"
        )
        draw.rectangle([0, 0, pil_img.width, banner_h], fill=banner_color)
        draw.text((12, 10), banner_text, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def summarize_detections(detections: list[Detection]) -> dict:
    """Compute counts and structured tabular rows for a list of detections."""
    persons = sum(1 for d in detections if d.class_name == "person")
    compliant = [d for d in detections if d.category == "Compliant"]
    violations = [d for d in detections if d.category == "Violation"]

    rows = [
        {
            "Class": d.class_name,
            "Category": d.category,
            "Confidence": round(d.confidence, 4),
            "x1": int(round(d.xyxy[0])),
            "y1": int(round(d.xyxy[1])),
            "x2": int(round(d.xyxy[2])),
            "y2": int(round(d.xyxy[3])),
        }
        for d in detections
    ]

    return {
        "total_detections": len(detections),
        "persons": persons,
        "compliant_count": len(compliant),
        "violation_count": len(violations),
        "violation_types": sorted({d.class_name for d in violations}),
        "is_compliant": len(violations) == 0,
        "dataframe": pd.DataFrame(rows),
    }


def process_video(
    video_path: str | Path,
    output_path: str | Path,
    detector_fn: Callable[[np.ndarray], list[Detection]],
    frame_stride: int = 3,
    max_frames: int = 150,
    allowed_classes: set[str] | None = None,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int = 3,
    progress_cb: Callable[[int, int, np.ndarray, dict], None] | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Run PPE detection across sampled video frames and write an annotated MP4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = max(1.0, fps / max(1, frame_stride))
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )

    log_rows: list[dict] = []
    frame_idx = 0
    processed = 0
    estimated_samples = (
        min(max_frames, max(1, total_frames // max(1, frame_stride)))
        if total_frames > 0
        else max_frames
    )

    try:
        while processed < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = detector_fn(frame_rgb)
            annotated_rgb = draw_annotations(
                frame_rgb,
                detections,
                allowed_classes=allowed_classes,
                show_labels=show_labels,
                show_conf=show_conf,
                line_width=line_width,
            )
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

            summary = summarize_detections(detections)
            timestamp_s = round(frame_idx / fps, 2)
            log_rows.append(
                {
                    "Frame": frame_idx,
                    "Time (s)": timestamp_s,
                    "Persons": summary["persons"],
                    "Compliant": summary["compliant_count"],
                    "Violations": summary["violation_count"],
                    "Violation Types": ", ".join(summary["violation_types"]) or "None",
                    "Status": "Compliant" if summary["is_compliant"] else "Violation",
                }
            )
            processed += 1
            if progress_cb is not None:
                progress_cb(processed, estimated_samples, annotated_rgb, summary)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return out_path, pd.DataFrame(log_rows)


def scan_dataset_stats(root: Path = PROJECT_ROOT) -> tuple[dict[str, int], pd.DataFrame]:
    """Count images and YOLO label class occurrences across train/val/test splits."""
    splits = ["train", "val", "test"]
    image_counts: dict[str, int] = {}
    class_rows: list[dict] = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in splits:
        img_dir = root / "dataset" / "images" / split
        lbl_dir = root / "dataset" / "labels" / split
        images = [p for p in img_dir.glob("*") if p.suffix.lower() in img_exts] if img_dir.exists() else []
        image_counts[split] = len(images)

        counts = {name: 0 for name in CLASS_NAMES}
        if lbl_dir.exists():
            for txt_file in lbl_dir.glob("*.txt"):
                for line in txt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= cid < len(CLASS_NAMES):
                        counts[CLASS_NAMES[cid]] += 1
        for cname, cnt in counts.items():
            class_rows.append({"Split": split, "Class": cname, "Instances": cnt})

    return image_counts, pd.DataFrame(class_rows)


def load_training_runs(root: Path = PROJECT_ROOT) -> list[dict]:
    """Discover Ultralytics training runs containing results.csv or evaluation plots."""
    runs_dir = root / "runs" / "train"
    if not runs_dir.exists():
        return []

    found: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        results_csv = run_dir / "results.csv"
        df = None
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df.columns = [c.strip() for c in df.columns]
        plots = [
            run_dir / fname
            for fname in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png")
            if (run_dir / fname).exists()
        ]
        weights = sorted((run_dir / "weights").glob("*.pt")) if (run_dir / "weights").exists() else []
        found.append(
            {
                "name": run_dir.name,
                "path": run_dir,
                "metrics_df": df,
                "plots": plots,
                "weights": weights,
            }
        )
    return found
