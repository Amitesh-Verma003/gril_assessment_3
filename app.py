"""
app.py  —  PPE Compliance Violation Detector (Gradio Web Demo)
--------------------------------------------------------------
Run in Google Colab AFTER training:

    !python app.py

This generates a PUBLIC shareable link you can send to your mentor.
They open it in any browser and upload a photo to get instant results.

Or launch with a custom share tunnel:
    demo.launch(share=True)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from ultralytics import YOLO

# ── Class metadata ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    "person", "hardhat", "hardhat_wrong", "no_hardhat",
    "vest",   "vest_wrong",   "no_vest",
    "gloves", "no_gloves",
]

VIOLATION_IDS = {2, 3, 5, 6, 8}

COMPLIANT_IDS = {1, 4, 7}          # hardhat, vest, gloves

# BGR colours (used by OpenCV)
BOX_COLORS_BGR = {
    0: (0,  165, 255),   # person       — orange
    1: (0,  200,   0),   # hardhat      — green
    2: (0,   0,  220),   # hardhat_wrong— blue-red
    3: (0,   0,  220),   # no_hardhat   — red
    4: (0,  200,   0),   # vest         — green
    5: (50,  50,  220),  # vest_wrong   — blue-red
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

# ── Load model ──────────────────────────────────────────────────────────────
# Default path — change if your weights are elsewhere
DEFAULT_WEIGHTS = "runs/train/ppe_yolov8n/weights/best.pt"

_model_cache: dict = {}

def load_model(weights_path: str) -> YOLO:
    if weights_path not in _model_cache:
        print(f"[INFO] Loading model: {weights_path}")
        _model_cache[weights_path] = YOLO(weights_path)
    return _model_cache[weights_path]


# ── Core inference + annotation ─────────────────────────────────────────────
def run_inference(image_rgb: np.ndarray,
                  weights_path: str,
                  conf_thresh: float,
                  iou_thresh: float,
                  show_person: bool) -> tuple[np.ndarray, str, str]:
    """
    Returns
    -------
    annotated_image_rgb : np.ndarray   drawn result (RGB, for Gradio)
    summary_text        : str          plain-text violation report
    status_emoji        : str          overall status label
    """
    model  = load_model(weights_path)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source  = img_bgr,
        conf    = conf_thresh,
        iou     = iou_thresh,
        imgsz   = 640,
        verbose = False,
    )[0]

    out = img_bgr.copy()
    h, w = out.shape[:2]

    detected_violations : list[str] = []
    detected_compliant  : list[str] = []
    person_count        : int       = 0
    rows = []   # for the text report

    boxes   = results.boxes.xyxy.cpu().numpy()
    cls_ids = results.boxes.cls.int().cpu().tolist()
    confs   = results.boxes.conf.cpu().tolist()

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        if cls_id == 0:
            person_count += 1
            if not show_person:
                continue        # skip drawing person box if toggled off

        x1, y1, x2, y2 = map(int, box)
        color = BOX_COLORS_BGR.get(cls_id, (180, 180, 180))
        label = f"{CLASS_NAMES[cls_id]}  {conf:.2f}"

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if cls_id in VIOLATION_IDS:
            detected_violations.append(VIOLATION_LABELS[cls_id])
            rows.append(f"  🔴 {VIOLATION_LABELS[cls_id]}  (conf={conf:.2f})")
        elif cls_id in COMPLIANT_IDS:
            detected_compliant.append(CLASS_NAMES[cls_id])
            rows.append(f"  🟢 {CLASS_NAMES[cls_id]}  (conf={conf:.2f})")

    # ── Top status banner ───────────────────────────────────────────────────
    if detected_violations:
        banner_text  = f"⚠  PPE VIOLATION  |  {len(detected_violations)} issue(s) found"
        banner_color = (0, 0, 180)        # red in BGR
        status_emoji = "🔴 VIOLATION"
    else:
        banner_text  = "✅  ALL PPE COMPLIANT"
        banner_color = (0, 150, 0)
        status_emoji = "🟢 COMPLIANT"

    cv2.rectangle(out, (0, 0), (w, 36), banner_color, -1)
    cv2.putText(out, banner_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Info sidebar (bottom strip) ─────────────────────────────────────────
    info = f"Persons: {person_count}   Violations: {len(detected_violations)}   Compliant items: {len(detected_compliant)}"
    cv2.rectangle(out, (0, h - 28), (w, h), (40, 40, 40), -1)
    cv2.putText(out, info, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # ── Plain-text report ───────────────────────────────────────────────────
    if rows:
        report = (
            f"👤 Persons detected : {person_count}\n"
            f"📋 Detections       : {len(boxes)}\n\n"
            "DETAILS:\n" + "\n".join(rows)
        )
    else:
        report = f"👤 Persons detected: {person_count}\n📋 No PPE items detected."

    return out_rgb, report, status_emoji


# ── Gradio UI ───────────────────────────────────────────────────────────────
def predict(image, weights_path, conf, iou, show_person):
    if image is None:
        return None, "⬆ Upload an image first.", ""
    try:
        ann, report, status = run_inference(
            image_rgb    = image,
            weights_path = weights_path.strip() or DEFAULT_WEIGHTS,
            conf_thresh  = conf,
            iou_thresh   = iou,
            show_person  = show_person,
        )
        return ann, report, status
    except FileNotFoundError as e:
        return None, f"❌ Model not found:\n{e}\n\nSet the correct weights path.", "❌ Error"
    except Exception as e:
        return None, f"❌ Error during inference:\n{e}", "❌ Error"


# ── Build Gradio interface ───────────────────────────────────────────────────
with gr.Blocks(title="PPE Violation Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🦺 PPE Compliance Violation Detector
    **YOLOv8 · 9 Classes · CCTV Workplace Safety**

    Upload any workplace / construction site image and the model will instantly:
    - Draw **bounding boxes** around each person and PPE item
    - Flag **violations** (missing / wrongly worn helmet, vest, gloves) in 🔴
    - Show **compliant** items in 🟢
    - Display an overall banner: ✅ COMPLIANT or ⚠ VIOLATION
    """)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            inp_image = gr.Image(
                label   = "📸 Upload Image",
                type    = "numpy",
                sources = ["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Settings", open=False):
                inp_weights = gr.Textbox(
                    label       = "Model weights path",
                    value       = DEFAULT_WEIGHTS,
                    placeholder = "runs/train/ppe_yolov8n/weights/best.pt",
                )
                inp_conf = gr.Slider(
                    minimum = 0.1, maximum = 0.9, value = 0.4, step = 0.05,
                    label   = "Confidence threshold",
                )
                inp_iou = gr.Slider(
                    minimum = 0.1, maximum = 0.9, value = 0.45, step = 0.05,
                    label   = "IoU threshold (NMS)",
                )
                inp_person = gr.Checkbox(
                    label = "Show 'person' bounding box",
                    value = True,
                )

            btn_run = gr.Button("🔍 Detect Violations", variant="primary", size="lg")

        # ── Right column: outputs ─────────────────────────────────────────
        with gr.Column(scale=2):
            out_status = gr.Label(label="Overall Status", num_top_classes=1)
            out_image  = gr.Image(label="📦 Annotated Result", type="numpy")
            out_report = gr.Textbox(
                label = "📋 Detection Report",
                lines = 10,
            )

    # ── Examples (use real test images if available) ──────────────────────
    gr.Markdown("### 📷 Or try an example image from the test set")
    example_imgs = sorted(Path("dataset/images/test").glob("*.*"))[:4]
    if example_imgs:
        gr.Examples(
            examples   = [[str(p)] for p in example_imgs],
            inputs     = [inp_image],
            label      = "Test set examples",
        )

    # ── Legend ────────────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    ### 🏷️ Colour Legend
    | Colour | Meaning |
    |--------|---------|
    | 🟩 Green box | PPE item present & correct |
    | 🟥 Red box | PPE missing (`no_hardhat`, `no_vest`, `no_gloves`) |
    | 🟦 Blue box | PPE worn incorrectly (`hardhat_wrong`, `vest_wrong`) |
    | 🟧 Orange box | Person anchor |
    """)

    btn_run.click(
        fn      = predict,
        inputs  = [inp_image, inp_weights, inp_conf, inp_iou, inp_person],
        outputs = [out_image, out_report, out_status],
    )

    # Also trigger on image upload
    inp_image.change(
        fn      = predict,
        inputs  = [inp_image, inp_weights, inp_conf, inp_iou, inp_person],
        outputs = [out_image, out_report, out_status],
    )


if __name__ == "__main__":
    demo.launch(
        share          = True,    # generates public gradio.live URL
        server_name    = "0.0.0.0",
        server_port    = 7860,
        show_error     = True,
        inbrowser      = False,   # set True if running locally
    )
