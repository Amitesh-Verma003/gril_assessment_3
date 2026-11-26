"""
app.py  —  PPE Compliance Violation Detector  (Local / Colab)
=============================================================
Three modes in one app:

  TAB 1 — Upload Image        : single image → annotated result
  TAB 2 — Live Webcam         : real-time feed from laptop camera
  TAB 3 — Upload Video        : .mp4 / .avi → annotated output video

Run locally:
    python app.py
    → opens at http://localhost:7860

Run in Colab (public link):
    python app.py --share
    → prints a public gradio.live URL
"""

import argparse
import sys
import os
import time
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# ── Class metadata ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    "person",        # 0
    "hardhat",       # 1
    "hardhat_wrong", # 2
    "no_hardhat",    # 3
    "vest",          # 4
    "vest_wrong",    # 5
    "no_vest",       # 6
    "gloves",        # 7
    "no_gloves",     # 8
]

VIOLATION_IDS = {2, 3, 5, 6, 8}
COMPLIANT_IDS = {1, 4, 7}

# BGR colour per class
BOX_COLORS = {
    0: (0,  165, 255),   # person       — orange
    1: (0,  200,   0),   # hardhat      — green
    2: (30,  30, 230),   # hardhat_wrong— red-blue
    3: (0,   0,  220),   # no_hardhat   — red
    4: (0,  200,   0),   # vest         — green
    5: (30,  30, 230),   # vest_wrong   — red-blue
    6: (0,   0,  220),   # no_vest      — red
    7: (0,  200,   0),   # gloves       — green
    8: (0,   0,  220),   # no_gloves    — red
}

VIOLATION_LABELS = {
    2: "⚠ Hardhat worn wrong",
    3: "❌ No hardhat",
    5: "⚠ Vest worn wrong",
    6: "❌ No vest",
    8: "❌ No gloves",
}

# Default weights path — override via CLI or the UI
DEFAULT_WEIGHTS = "weights/best.onnx"

# ── Model loader (cached) ───────────────────────────────────────────────────
_model_cache: dict = {}

def load_model(path: str) -> YOLO:
    path = path.strip()
    if path not in _model_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Weights not found: {path}\n"
                "Copy best.onnx to weights/best.onnx  (already present ✅)"
            )
        print(f"[INFO] Loading model: {path}")
        _model_cache[path] = YOLO(path)
    return _model_cache[path]


# ── Core annotator ──────────────────────────────────────────────────────────
def annotate_frame(
    frame_bgr: np.ndarray,
    model: YOLO,
    conf: float,
    iou: float,
    show_person: bool,
    fps: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Run YOLOv8 on one BGR frame.
    Returns (annotated_BGR, list_of_violation_strings).
    """
    h, w = frame_bgr.shape[:2]
    results = model.predict(
        source  = frame_bgr,
        conf    = conf,
        iou     = iou,
        imgsz   = 640,
        verbose = False,
        device  = "cpu",          # safe default for laptops without NVIDIA GPU
    )[0]

    out        = frame_bgr.copy()
    violations : list[str] = []
    compliant  : list[str] = []

    boxes   = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.int().cpu().tolist()
    confs   = results.boxes.conf.cpu().tolist()

    for box, cls_id, cf in zip(boxes, cls_ids, confs):
        if cls_id == 0 and not show_person:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = BOX_COLORS.get(cls_id, (180, 180, 180))
        label = f"{CLASS_NAMES[cls_id]}  {cf:.2f}"

        # bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # label pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        if cls_id in VIOLATION_IDS:
            violations.append(VIOLATION_LABELS[cls_id])
        elif cls_id in COMPLIANT_IDS:
            compliant.append(CLASS_NAMES[cls_id])

    # ── Top status banner ───────────────────────────────────────────────────
    if violations:
        banner = f"  VIOLATION  |  {len(violations)} issue(s)"
        bc     = (0, 0, 170)
    else:
        banner = "  ALL PPE COMPLIANT"
        bc     = (0, 140, 0)

    cv2.rectangle(out, (0, 0), (w, 34), bc, -1)
    cv2.putText(out, banner, (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    # ── FPS overlay (webcam / video) ────────────────────────────────────────
    if fps is not None:
        fps_txt = f"FPS: {fps:.1f}"
        cv2.putText(out, fps_txt, (w - 110, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2, cv2.LINE_AA)

    # ── Bottom info strip ───────────────────────────────────────────────────
    info = (f"Persons: {cls_ids.count(0)}  "
            f"Violations: {len(violations)}  "
            f"Compliant items: {len(compliant)}")
    cv2.rectangle(out, (0, h - 26), (w, h), (30, 30, 30), -1)
    cv2.putText(out, info, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    return out, violations


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE
# ═══════════════════════════════════════════════════════════════════════════
def predict_image(image_rgb, weights, conf, iou, show_person):
    if image_rgb is None:
        return None, "⬆ Upload an image first.", "—"
    try:
        model  = load_model(weights)
        bgr    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        out, v = annotate_frame(bgr, model, conf, iou, show_person)
        rgb    = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        report = (
            f"Violations found ({len(v)}):\n" + "\n".join(f"  {x}" for x in set(v))
            if v else "✅ No violations detected."
        )
        status = "🔴 VIOLATION" if v else "🟢 COMPLIANT"
        return rgb, report, status
    except Exception as e:
        return None, f"❌ {e}", "❌ Error"


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — LIVE WEBCAM  (Gradio streaming)
# ═══════════════════════════════════════════════════════════════════════════
# Gradio calls this function on EVERY streamed frame from the webcam
_webcam_fps_tracker: dict = {"t": time.time(), "count": 0, "fps": 0.0}

def predict_webcam(frame_rgb, weights, conf, iou, show_person):
    """Called per-frame by Gradio's .stream() — must be fast."""
    if frame_rgb is None:
        return None

    # FPS calculation
    t = _webcam_fps_tracker
    t["count"] += 1
    elapsed = time.time() - t["t"]
    if elapsed >= 1.0:
        t["fps"]   = t["count"] / elapsed
        t["count"] = 0
        t["t"]     = time.time()

    try:
        model  = load_model(weights)
        bgr    = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out, _ = annotate_frame(bgr, model, conf, iou, show_person,
                                fps=t["fps"])
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except Exception as e:
        # write error on frame so user sees it
        err = frame_rgb.copy()
        cv2.putText(err, f"ERROR: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return err


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — VIDEO FILE
# ═══════════════════════════════════════════════════════════════════════════
def process_video(video_path, weights, conf, iou, show_person,
                  max_frames: int = 500):
    """
    Read every frame, annotate it, write to a temp output .mp4.
    Returns (output_video_path, report_string).
    max_frames caps processing so the UI doesn't hang for very long videos.
    """
    if video_path is None:
        return None, "⬆ Upload a video file first."

    try:
        model = load_model(weights)
    except Exception as e:
        return None, f"❌ {e}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(tempfile.gettempdir(), "ppe_output.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    frame_count     = 0
    total_violations: list[str] = []
    t0 = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        fps_now = frame_count / max(time.time() - t0, 1e-6)
        out, v  = annotate_frame(frame, model, conf, iou, show_person,
                                 fps=fps_now)
        writer.write(out)
        total_violations.extend(v)
        frame_count += 1

    cap.release()
    writer.release()

    # Build report
    from collections import Counter
    vc = Counter(total_violations)
    capped = frame_count < total
    report = (
        f"📹 Frames processed : {frame_count}{' (capped)' if capped else ''} / {total}\n"
        f"⏱ Time taken       : {time.time()-t0:.1f}s\n\n"
    )
    if vc:
        report += "Violation frequency:\n"
        report += "\n".join(f"  {lbl}: {cnt} frame(s)" for lbl, cnt in vc.most_common())
    else:
        report += "✅ No violations detected across all frames."

    return out_path, report


# ═══════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════
def build_settings_sidebar(tab_name: str):
    """Reusable settings block used in all three tabs."""
    with gr.Accordion("⚙️ Settings", open=True):
        weights = gr.Textbox(
            label       = "Path to weights",
            value       = DEFAULT_WEIGHTS,
            info        = "weights/best.onnx (already present ✅)",
        )
        conf = gr.Slider(0.05, 0.9, value=0.20, step=0.05,
                         label="Confidence threshold (Recommended: 0.20 for ONNX on CPU)")
        iou  = gr.Slider(0.1, 0.9, value=0.45, step=0.05,
                         label="IoU threshold (NMS)")
        person = gr.Checkbox(value=True, label="Draw 'person' box")
    return weights, conf, iou, person


with gr.Blocks(title="PPE Violation Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🦺 PPE Compliance Violation Detector — Local App
    **YOLOv8 · 9 Classes · Hardhat / Vest / Gloves**

    | 🖼 Tab 1 | 📷 Tab 2 | 🎬 Tab 3 |
    |----------|----------|----------|
    | Upload an image | Live webcam feed | Upload a video file |

    > Copy `best.onnx` from Colab to `weights/best.onnx` before running.
    """)

    # ── TAB 1 : IMAGE ───────────────────────────────────────────────────────
    with gr.Tab("🖼️  Image Upload"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    label   = "📸 Upload or paste image",
                    type    = "numpy",
                    sources = ["upload", "clipboard"],
                )
                w1, c1, i1, p1 = build_settings_sidebar("img")
                btn_img = gr.Button("🔍 Detect", variant="primary", size="lg")

            with gr.Column(scale=2):
                img_status = gr.Label(label="Overall Status")
                img_output = gr.Image(label="Annotated Result", type="numpy")
                img_report = gr.Textbox(label="Detection Report", lines=8)

        btn_img.click(
            fn      = predict_image,
            inputs  = [img_input, w1, c1, i1, p1],
            outputs = [img_output, img_report, img_status],
        )

    # ── TAB 2 : LIVE WEBCAM ─────────────────────────────────────────────────
    with gr.Tab("📷  Webcam Snapshot / Feed"):
        gr.Markdown("""
        > **Tip for Laptop Camera:** Snap a photo or record a short clip from your webcam and run detection.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                webcam_in = gr.Image(
                    label     = "📷 Webcam",
                    sources   = ["webcam"],
                    type      = "numpy",
                )
                w2, c2, i2, p2 = build_settings_sidebar("cam")
                btn_cam = gr.Button("🔍 Run Detection on Webcam Photo", variant="primary", size="lg")

            with gr.Column(scale=2):
                cam_status = gr.Label(label="Overall Status")
                webcam_out = gr.Image(
                    label = "🔍 Annotated Detection Output",
                    type  = "numpy",
                )
                cam_report = gr.Textbox(label="Detection Report", lines=8)

        btn_cam.click(
            fn      = predict_image,
            inputs  = [webcam_in, w2, c2, i2, p2],
            outputs = [webcam_out, cam_report, cam_status],
        )

    # ── TAB 3 : VIDEO FILE ──────────────────────────────────────────────────
    with gr.Tab("🎬  Video File"):
        gr.Markdown("""
        > Upload an `.mp4` or `.avi` file.  
        > **Processing is done frame-by-frame on CPU — allow ~1–2 min for a 1-min clip.**  
        > Output is saved as an annotated `.mp4` you can download.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                vid_input  = gr.Video(label="📂 Upload Video", sources=["upload"])
                max_frames = gr.Slider(50, 1000, value=300, step=50,
                                       label="Max frames to process (keep low on CPU)")
                w3, c3, i3, p3 = build_settings_sidebar("vid")
                btn_vid = gr.Button("▶ Process Video", variant="primary")

            with gr.Column(scale=2):
                vid_output = gr.Video(label="📦 Annotated Output Video")
                vid_report = gr.Textbox(label="📋 Video Report", lines=10)

        btn_vid.click(
            fn      = lambda vp, w, c, i, p, mf: process_video(vp, w, c, i, p, int(mf)),
            inputs  = [vid_input, w3, c3, i3, p3, max_frames],
            outputs = [vid_output, vid_report],
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    with gr.Accordion("🏷️ Colour Legend", open=False):
        gr.Markdown("""
        | Box colour | Class |
        |-----------|-------|
        | 🟩 Green  | `hardhat` · `vest` · `gloves` — PPE worn correctly |
        | 🟥 Red    | `no_hardhat` · `no_vest` · `no_gloves` — PPE missing |
        | 🟦 Blue   | `hardhat_wrong` · `vest_wrong` — PPE worn incorrectly |
        | 🟧 Orange | `person` anchor box |
        """)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to weights (default: weights/best.onnx)")
    parser.add_argument("--port",    type=int, default=7860)
    parser.add_argument("--share",   action="store_true",
                        help="Generate public gradio.live URL (for Colab / mentor demo)")
    args = parser.parse_args()

    # pre-load model at startup so first inference is instant
    if os.path.exists(args.weights):
        DEFAULT_WEIGHTS = args.weights
        try:
            load_model(args.weights)
            print(f"[INFO] Model pre-loaded: {args.weights}")
        except Exception as e:
            print(f"[WARN] Could not pre-load model: {e}")
    else:
        print(f"[WARN] Weights not found at {args.weights}")
        print("       Expected: weights/best.onnx")

    demo.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
        inbrowser   = True,         # auto-open browser tab locally
        show_error  = True,
    )
