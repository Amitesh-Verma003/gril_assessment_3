"""
inference.py
------------
Shared helpers for PPE compliance model loading, inference, annotation,
video processing, and dataset/training inspection used by the Streamlit UI.
"""

from __future__ import annotations

import ast
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
    source: str = "model"  # "model" | "compliance_rule"


def is_ultralytics_available() -> bool:
    """Return True if ultralytics can be imported in the current environment."""
    return importlib.util.find_spec("ultralytics") is not None


def is_onnxruntime_available() -> bool:
    """Return True if onnxruntime can be imported in the current environment."""
    return importlib.util.find_spec("onnxruntime") is not None


def classify_id(class_id: int) -> str:
    """Map a class index to its compliance category."""
    if class_id in VIOLATION_IDS:
        return "Violation"
    if class_id in COMPLIANT_IDS:
        return "Compliant"
    return "Anchor"


def discover_weights(root: Path = PROJECT_ROOT) -> list[Path]:
    """Find available .onnx and .pt checkpoints in weights/, models/weights/, runs/, and root."""
    candidates: list[Path] = []
    search_dirs = [root / "weights", root / "models" / "weights", root / "runs", root]
    seen: set[Path] = set()
    for base in search_dirs:
        if not base.exists():
            continue
        for ext in ("*.onnx", "*.pt"):
            pattern = ext if base == root else f"**/{ext}"
            for weight_path in sorted(base.glob(pattern)):
                resolved = weight_path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(weight_path)
    return candidates


SUPPORTED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif", ".ico", ".ppm", ".pgm", ".pbm", ".tga",
}


def decode_image_bytes(raw_bytes: bytes, filename: str = "") -> np.ndarray:
    """Decode uploaded image bytes (JPEG, PNG, WEBP, BMP, TIFF, GIF, etc.) into RGB numpy array."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            if getattr(pil_img, "is_animated", False):
                pil_img.seek(0)
            transposed = ImageOps.exif_transpose(pil_img)
            return np.array(transposed.convert("RGB"))
    except Exception:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        label = f" ({filename})" if filename else ""
        raise ValueError(f"Unsupported or corrupted image file{label}.")


def discover_sample_images(root: Path = PROJECT_ROOT, limit: int = 30) -> list[Path]:
    """Find sample images in dataset/images/test, val, or outputs if present."""
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
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                found.append(p)
                if len(found) >= limit:
                    return found
    return found


def load_dataset_config(root: Path = PROJECT_ROOT) -> dict:
    """Load dataset/ppe_dataset.yaml or root dataset.yaml if present."""
    for cfg_path in (root / "dataset" / "ppe_dataset.yaml", root / "dataset.yaml"):
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {"nc": len(CLASS_NAMES), "names": dict(enumerate(CLASS_NAMES))}


class OnnxYoloModel:
    """Lightweight YOLOv8 ONNX inference wrapper using onnxruntime."""

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device != "cpu" and "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # typically [1, 3, 640, 640]

        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw_names = meta.get("names")
        if raw_names:
            parsed = ast.literal_eval(raw_names)
            self.names = {int(k): str(v) for k, v in parsed.items()}
        else:
            self.names = dict(enumerate(CLASS_NAMES))

    def _target_hw(self, fallback_imgsz: int) -> tuple[int, int]:
        h = self.input_shape[2] if len(self.input_shape) >= 4 else fallback_imgsz
        w = self.input_shape[3] if len(self.input_shape) >= 4 else fallback_imgsz
        th = int(h) if isinstance(h, (int, np.integer)) and h > 0 else fallback_imgsz
        tw = int(w) if isinstance(w, (int, np.integer)) and w > 0 else fallback_imgsz
        return th, tw

    def predict_rgb(
        self,
        image_rgb: np.ndarray,
        conf: float = 0.40,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> list[Detection]:
        orig_h, orig_w = image_rgb.shape[:2]
        target_h, target_w = self._target_hw(imgsz)

        scale = min(target_h / orig_h, target_w / orig_w)
        unpad_w = max(1, int(round(orig_w * scale)))
        unpad_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(image_rgb, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

        dw = (target_w - unpad_w) / 2.0
        dh = (target_h - unpad_h) / 2.0
        top = int(round(dh - 0.1))
        left = int(round(dw - 0.1))
        bottom = max(0, target_h - unpad_h - top)
        right = max(0, target_w - unpad_w - left)

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        blob = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        raw_out = self.session.run(None, {self.input_name: blob})[0]
        preds = np.squeeze(raw_out, axis=0)
        # Standard YOLOv8 ONNX output is (4 + nc, N); transpose to (N, 4 + nc)
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
            preds = preds.T

        if preds.size == 0 or preds.shape[1] < 5:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        # Multi-label decoding: keep all (anchor, class) pairs above threshold so
        # a high-scoring 'person' anchor does not hide overlapping PPE classes.
        anchor_idx, class_ids = np.where(class_scores >= conf)
        if len(anchor_idx) == 0:
            return []

        confidences = class_scores[anchor_idx, class_ids]
        if len(confidences) > 3000:
            top_k = np.argsort(confidences)[-3000:]
            anchor_idx = anchor_idx[top_k]
            class_ids = class_ids[top_k]
            confidences = confidences[top_k]

        boxes_xywh = boxes_xywh[anchor_idx]

        # Convert center-xywh (letterboxed) -> original image xyxy
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = np.clip((cx - bw / 2.0 - left) / scale, 0, orig_w)
        y1 = np.clip((cy - bh / 2.0 - top) / scale, 0, orig_h)
        x2 = np.clip((cx + bw / 2.0 - left) / scale, 0, orig_w)
        y2 = np.clip((cy + bh / 2.0 - top) / scale, 0, orig_h)

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return []
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        class_ids = class_ids[valid]
        confidences = confidences[valid]

        # Class-aware NMS using coordinate offsets per class
        max_dim = float(max(orig_w, orig_h, 1))
        nms_boxes = []
        for bx1, by1, bx2, by2, cid in zip(x1, y1, x2, y2, class_ids):
            offset = float(cid) * max_dim
            nms_boxes.append([float(bx1 + offset), float(by1 + offset), float(bx2 - bx1), float(by2 - by1)])

        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), score_threshold=conf, nms_threshold=iou)
        if len(indices) == 0:
            return []

        flat_indices = np.array(indices).reshape(-1).tolist()
        detections: list[Detection] = []
        for idx in flat_indices:
            cls_id = int(class_ids[idx])
            name = self.names.get(cls_id, CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
            resolved_id = CLASS_NAMES.index(name) if name in CLASS_NAMES else cls_id
            detections.append(
                Detection(
                    xyxy=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                    class_id=resolved_id,
                    class_name=name,
                    confidence=float(confidences[idx]),
                    category=classify_id(resolved_id) if name in CLASS_NAMES else "Anchor",
                )
            )
        return detections


def load_yolo_model(weights_source: str, device: str = "cpu"):
    """Load an ONNX Runtime YOLOv8 model or Ultralytics .pt model."""
    if str(weights_source).lower().endswith(".onnx"):
        if is_onnxruntime_available():
            return OnnxYoloModel(weights_source, device=device)
        if not is_ultralytics_available():
            raise RuntimeError(
                "Neither 'onnxruntime' nor 'ultralytics' is installed to run `.onnx` weights."
            )

    if not is_ultralytics_available():
        raise RuntimeError(
            "The 'ultralytics' package is not installed in this Python environment. "
            "Use an `.onnx` checkpoint with `onnxruntime`, install `ultralytics`, or enable Demo Mode."
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
    if hasattr(model, "predict_rgb"):
        return model.predict_rgb(image_rgb, conf=conf, iou=iou, imgsz=imgsz)

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
    font_size = max(13, min(26, int(min(pil_img.width, pil_img.height) * 0.028)))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()

    filtered = [
        d for d in detections if allowed_classes is None or d.class_name in allowed_classes
    ]
    has_violation = any(d.category == "Violation" for d in detections)

    if show_banner:
        banner_h = max(32, font_size + 16)
        banner_color = (185, 28, 28) if has_violation else (21, 128, 61)
        banner_text = (
            "PPE VIOLATION DETECTED" if has_violation else "PPE COMPLIANT"
        )
        draw.rectangle([0, 0, pil_img.width, banner_h], fill=banner_color)
        draw.text((12, 8), banner_text, fill=(255, 255, 255), font=font)

    for det in filtered:
        x1, y1, x2, y2 = det.xyxy
        color = CLASS_COLORS_RGB.get(det.class_id, (148, 163, 184))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if show_labels:
            tag = " [rule]" if det.source == "compliance_rule" else ""
            label = (
                f"{det.class_name}{tag} {det.confidence:.2f}"
                if show_conf
                else f"{det.class_name}{tag}"
            )
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = 4
            label_left = max(0.0, min(x1, float(pil_img.width - tw - pad * 2)))
            label_top = max(0.0, y1 - th - pad * 2)
            draw.rectangle(
                [label_left, label_top, label_left + tw + pad * 2, label_top + th + pad * 2],
                fill=color,
            )
            text_color = (15, 23, 42) if det.class_id in (0, 2, 5) else (255, 255, 255)
            draw.text((label_left + pad, label_top + pad), label, fill=text_color, font=font)

    return np.array(pil_img)


def _belongs_to_person(gear_box: tuple[float, float, float, float], person_box: tuple[float, float, float, float]) -> bool:
    """Return True if a PPE box overlaps significantly with an expanded person bounding box."""
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    # Expand slightly above head and sides to capture hardhats resting on top of the person box
    ex1, ey1, ex2, ey2 = px1 - 0.10 * pw, py1 - 0.15 * ph, px2 + 0.10 * pw, py2 + 0.05 * ph

    gx1, gy1, gx2, gy2 = gear_box
    ix1, iy1 = max(ex1, gx1), max(ey1, gy1)
    ix2, iy2 = min(ex2, gx2), min(ey2, gy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    gear_area = max(1.0, (gx2 - gx1) * (gy2 - gy1))
    return (inter_area / gear_area) >= 0.25


def _should_merge_persons(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Return True if two person boxes represent nested fragments or upper/lower halves of one human."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    wa, ha = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    wb, hb = max(1.0, bx2 - bx1), max(1.0, by2 - by1)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 > ix1 and iy2 > iy1:
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        if inter_area / min(wa * ha, wb * hb) >= 0.55:
            return True

    # Vertical split check: strong horizontal alignment, similar width, overlapping or touching vertically
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    x_ratio = x_overlap / min(wa, wb)
    w_ratio = min(wa, wb) / max(wa, wb)
    y_gap = max(ay1, by1) - min(ay2, by2)
    return x_ratio >= 0.60 and w_ratio >= 0.55 and y_gap <= 0.05 * max(ha, hb)


def merge_split_person_boxes(detections: list[Detection]) -> list[Detection]:
    """Merge duplicate nested or split upper/lower-body person detections into one person box."""
    persons = sorted(
        [d for d in detections if d.class_name == "person"],
        key=lambda d: d.confidence,
        reverse=True,
    )
    others = [d for d in detections if d.class_name != "person"]
    if len(persons) <= 1:
        return detections

    merged: list[Detection] = []
    for p in persons:
        matched_idx = None
        for idx, m in enumerate(merged):
            if _should_merge_persons(m.xyxy, p.xyxy):
                matched_idx = idx
                break
        if matched_idx is None:
            merged.append(p)
        else:
            m = merged[matched_idx]
            merged[matched_idx] = Detection(
                xyxy=(
                    min(m.xyxy[0], p.xyxy[0]),
                    min(m.xyxy[1], p.xyxy[1]),
                    max(m.xyxy[2], p.xyxy[2]),
                    max(m.xyxy[3], p.xyxy[3]),
                ),
                class_id=m.class_id,
                class_name=m.class_name,
                confidence=max(m.confidence, p.confidence),
                category=m.category,
                source=m.source,
            )

    return merged + others


def audit_person_compliance(
    detections: list[Detection],
    require_hardhat: bool = True,
    require_vest: bool = True,
) -> list[Detection]:
    """Flag detected persons who lack a visible compliant hardhat or safety vest."""
    detections = merge_split_person_boxes(detections)
    persons = [d for d in detections if d.class_name == "person"]
    if not persons:
        return detections

    head_items = [d for d in detections if d.class_name in {"hardhat", "hardhat_wrong", "no_hardhat"}]
    torso_items = [d for d in detections if d.class_name in {"vest", "vest_wrong", "no_vest"}]

    augmented = list(detections)
    for p in persons:
        px1, py1, px2, py2 = p.xyxy
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)

        if require_hardhat and not any(_belongs_to_person(g.xyxy, p.xyxy) for g in head_items):
            augmented.append(
                Detection(
                    xyxy=(px1 + 0.20 * pw, py1, px2 - 0.20 * pw, py1 + 0.28 * ph),
                    class_id=3,
                    class_name="no_hardhat",
                    confidence=p.confidence,
                    category="Violation",
                    source="compliance_rule",
                )
            )
        if require_vest and not any(_belongs_to_person(g.xyxy, p.xyxy) for g in torso_items):
            augmented.append(
                Detection(
                    xyxy=(px1 + 0.12 * pw, py1 + 0.30 * ph, px2 - 0.12 * pw, min(py2, py1 + 0.78 * ph)),
                    class_id=6,
                    class_name="no_vest",
                    confidence=p.confidence,
                    category="Violation",
                    source="compliance_rule",
                )
            )
    return augmented


def summarize_detections(detections: list[Detection]) -> dict:
    """Compute counts and structured tabular rows for a list of detections."""
    persons = sum(1 for d in detections if d.class_name == "person")
    compliant = [d for d in detections if d.category == "Compliant"]
    violations = [d for d in detections if d.category == "Violation"]

    rows = [
        {
            "Class": d.class_name,
            "Category": d.category,
            "Source": "Compliance Rule" if d.source == "compliance_rule" else "Model",
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


def discover_exported_plots(root: Path = PROJECT_ROOT) -> list[Path]:
    """Find exported evaluation/training plot images in plots/."""
    plots_dir = root / "plots"
    if not plots_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in sorted(plots_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

