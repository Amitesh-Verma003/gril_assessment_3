"""
app.py
------
Streamlit frontend for YOLOv8 PPE Compliance Violation Detection.

Run from project root:
    streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image

from utils.inference import (
    CLASS_COLORS_RGB,
    CLASS_NAMES,
    PROJECT_ROOT,
    Detection,
    audit_person_compliance,
    classify_id,
    decode_image_bytes,
    discover_exported_plots,
    discover_sample_images,
    discover_weights,
    draw_annotations,
    generate_demo_detections,
    is_onnxruntime_available,
    is_ultralytics_available,
    load_dataset_config,
    load_training_runs,
    load_yolo_model,
    process_video,
    run_image_inference,
    scan_dataset_stats,
    summarize_detections,
)

st.set_page_config(
    page_title="PPE Compliance Monitor — YOLOv8",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .status-banner {
        padding: 0.85rem 1.2rem;
        border-radius: 0.6rem;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .status-ok {
        background-color: rgba(21, 128, 61, 0.14);
        border: 1px solid rgba(34, 197, 94, 0.55);
        color: #16a34a;
    }
    .status-violation {
        background-color: rgba(185, 28, 28, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.55);
        color: #ef4444;
    }
    .legend-pill {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading PPE detection model...")
def get_cached_model(weights_path: str, device: str = "cpu"):
    """Cache the loaded ONNX or YOLO model across Streamlit reruns."""
    return load_yolo_model(weights_path, device=device)


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Encode an RGB numpy image to PNG bytes for download buttons."""
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


def create_synthetic_site_image(width: int = 960, height: int = 540) -> np.ndarray:
    """Create a clean placeholder worksite image for testing Demo Mode without local images."""
    img = Image.new("RGB", (width, height), (30, 41, 59))
    arr = np.array(img, dtype=np.uint8)
    # Subtle floor gradient
    arr[int(height * 0.65) :, :] = (51, 65, 85)
    # Two stylized worker silhouettes matching demo detection coordinates
    pil = Image.fromarray(arr)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil)
    # Worker 1 (compliant)
    draw.ellipse([int(0.22 * width), int(0.18 * height), int(0.33 * width), int(0.31 * height)], fill=(226, 232, 240))
    draw.rectangle([int(0.17 * width), int(0.33 * height), int(0.39 * width), int(0.66 * height)], fill=(234, 179, 8))
    draw.rectangle([int(0.19 * width), int(0.66 * height), int(0.37 * width), int(0.89 * height)], fill=(71, 85, 105))
    # Worker 2 (non-compliant)
    draw.ellipse([int(0.65 * width), int(0.20 * height), int(0.76 * width), int(0.33 * height)], fill=(226, 232, 240))
    draw.rectangle([int(0.60 * width), int(0.35 * height), int(0.81 * width), int(0.68 * height)], fill=(100, 116, 139))
    draw.rectangle([int(0.62 * width), int(0.68 * height), int(0.79 * width), int(0.90 * height)], fill=(71, 85, 105))
    return np.array(pil)


def render_sidebar() -> dict:
    """Render sidebar controls for model selection, thresholds, and display filters."""
    st.sidebar.title("⚙️ PPE Model Controls")

    ultra_ok = is_ultralytics_available()
    onnx_ok = is_onnxruntime_available()
    cuda_ok = torch.cuda.is_available()

    col_env1, col_env2 = st.sidebar.columns(2)
    col_env1.caption(f"ONNX: `{'Ready' if onnx_ok else 'Missing'}`")
    col_env2.caption(f"GPU: `{'CUDA' if cuda_ok else 'CPU only'}`")

    if not onnx_ok and not ultra_ok:
        st.sidebar.warning(
            "Neither `onnxruntime` nor `ultralytics` is installed. "
            "Install one of them or use **Demo Mode** below."
        )

    discovered = discover_weights()
    weight_options = [str(p.relative_to(PROJECT_ROOT)) for p in discovered]

    st.sidebar.subheader("1. Model Weights")
    source_mode = st.sidebar.radio(
        "Weights Source",
        options=["Project Checkpoints", "Upload Model (.onnx / .pt)", "Custom Path / Hub ID"],
        index=0,
    )

    selected_weights: str | None = None
    if source_mode == "Project Checkpoints":
        if weight_options:
            choice = st.sidebar.selectbox("Select checkpoint", weight_options)
            selected_weights = str(PROJECT_ROOT / choice)
        else:
            st.sidebar.info("No `.onnx` or `.pt` files found in `weights/`, `models/weights/`, or `runs/train/`.")
    elif source_mode == "Upload Model (.onnx / .pt)":
        uploaded_model = st.sidebar.file_uploader("Upload `.onnx` or `.pt` weights", type=["onnx", "pt"])
        if uploaded_model is not None:
            weights_dir = PROJECT_ROOT / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            dest_path = weights_dir / uploaded_model.name
            dest_path.write_bytes(uploaded_model.getvalue())
            selected_weights = str(dest_path)
            st.sidebar.success(f"Saved to `weights/{uploaded_model.name}`")
    else:
        default_custom = weight_options[0] if weight_options else "weights/best.onnx"
        custom_path = st.sidebar.text_input(
            "Weights path or Ultralytics model",
            value=default_custom,
        )
        if custom_path.strip():
            candidate = Path(custom_path.strip())
            selected_weights = str(candidate if candidate.is_absolute() else PROJECT_ROOT / candidate)

    is_onnx_selected = bool(selected_weights and selected_weights.lower().endswith(".onnx"))
    can_run_selected = (
        (onnx_ok or ultra_ok) if is_onnx_selected else ultra_ok
    )
    default_demo = selected_weights is None or not can_run_selected

    demo_mode = st.sidebar.toggle(
        "Demo Mode (Simulated Detections)",
        value=default_demo,
        help="Generates synthetic PPE detections so you can test the UI without trained weights.",
    )

    st.sidebar.subheader("2. Inference Parameters")
    conf = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU Threshold (NMS)", 0.10, 0.95, 0.45, 0.05)
    imgsz = st.sidebar.select_slider("Image Size (px)", options=[320, 480, 640, 960, 1280], value=640)
    device_options = ["auto", "cpu"] + (["0"] if cuda_ok else [])
    device = st.sidebar.selectbox("Device", options=device_options, index=0)

    st.sidebar.subheader("3. PPE Compliance Rules")
    require_hardhat = st.sidebar.checkbox(
        "Flag person missing Hardhat",
        value=True,
        help="When a person is detected without a visible hardhat box, flag no_hardhat.",
    )
    require_vest = st.sidebar.checkbox(
        "Flag person missing Safety Vest",
        value=True,
        help="When a person is detected without a visible vest box, flag no_vest.",
    )

    st.sidebar.subheader("4. Display & Class Filter")
    filter_preset = st.sidebar.selectbox(
        "Class Filter",
        options=["All Classes", "Violations Only", "Compliant Only", "Custom Selection"],
    )
    if filter_preset == "All Classes":
        allowed_classes = set(CLASS_NAMES)
    elif filter_preset == "Violations Only":
        allowed_classes = {c for i, c in enumerate(CLASS_NAMES) if classify_id(i) == "Violation"}
    elif filter_preset == "Compliant Only":
        allowed_classes = {c for i, c in enumerate(CLASS_NAMES) if classify_id(i) == "Compliant"}
    else:
        chosen = st.sidebar.multiselect("Visible Classes", options=CLASS_NAMES, default=CLASS_NAMES)
        allowed_classes = set(chosen)

    show_labels = st.sidebar.checkbox("Show Class Labels", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
    line_width = st.sidebar.slider("Box Thickness", 1, 6, 3)

    with st.sidebar.expander("🏷️ PPE Class Legend", expanded=False):
        pills = []
        for idx, name in enumerate(CLASS_NAMES):
            r, g, b = CLASS_COLORS_RGB[idx]
            cat = classify_id(idx)
            pills.append(
                f"<span class='legend-pill' style='background:rgba({r},{g},{b},0.16); "
                f"color:rgb({r},{g},{b});'>{idx}: {name} ({cat})</span>"
            )
        st.markdown("".join(pills), unsafe_allow_html=True)

    return {
        "demo_mode": demo_mode,
        "weights": selected_weights,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
        "require_hardhat": require_hardhat,
        "require_vest": require_vest,
        "allowed_classes": allowed_classes,
        "show_labels": show_labels,
        "show_conf": show_conf,
        "line_width": line_width,
        "ultra_ok": ultra_ok,
        "onnx_ok": onnx_ok,
    }


def make_detector(cfg: dict):
    """Return a callable (image_rgb) -> list[Detection] and an optional status error."""
    if cfg["demo_mode"]:
        return lambda img: generate_demo_detections(img, conf_threshold=cfg["conf"]), None

    if not cfg["weights"]:
        return None, "No model checkpoint selected. Select or upload `.onnx` / `.pt` weights, or enable Demo Mode."

    weights_str = cfg["weights"]
    is_onnx = weights_str.lower().endswith(".onnx")

    if is_onnx and not (cfg["onnx_ok"] or cfg["ultra_ok"]):
        return None, "Install `onnxruntime` (or `ultralytics`) to run `.onnx` weights, or enable Demo Mode."
    if not is_onnx and not cfg["ultra_ok"]:
        return None, "Ultralytics is required for `.pt` weights. Select an `.onnx` checkpoint, install `ultralytics`, or enable Demo Mode."

    weights_path = Path(weights_str)
    if (is_onnx or weights_path.is_absolute() or "/" in weights_str or "\\" in weights_str) and not weights_path.exists():
        return None, f"Weights file not found: `{weights_str}`"

    try:
        model = get_cached_model(weights_str, device=cfg["device"])
    except Exception as exc:
        return None, f"Failed to load model `{weights_str}`: {exc}"

    def _detect(img_rgb: np.ndarray) -> list[Detection]:
        dets = run_image_inference(
            model,
            img_rgb,
            conf=cfg["conf"],
            iou=cfg["iou"],
            imgsz=cfg["imgsz"],
            device=cfg["device"],
        )
        if cfg["require_hardhat"] or cfg["require_vest"]:
            dets = audit_person_compliance(
                dets,
                require_hardhat=cfg["require_hardhat"],
                require_vest=cfg["require_vest"],
            )
        return dets

    return _detect, None


def render_detection_result(
    name: str,
    image_rgb: np.ndarray,
    detections: list[Detection],
    cfg: dict,
    side_by_side: bool = True,
):
    """Display status banner, metrics, annotated image, and detection breakdown."""
    summary = summarize_detections(detections)
    annotated = draw_annotations(
        image_rgb,
        detections,
        allowed_classes=cfg["allowed_classes"],
        show_labels=cfg["show_labels"],
        show_conf=cfg["show_conf"],
        line_width=cfg["line_width"],
    )

    if summary["is_compliant"]:
        st.markdown(
            f"<div class='status-banner status-ok'>"
            f"<span>✅ PPE COMPLIANT — {name}</span>"
            f"<span>0 Violations Detected</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        violations_str = ", ".join(summary["violation_types"])
        st.markdown(
            f"<div class='status-banner status-violation'>"
            f"<span>⚠️ PPE VIOLATION DETECTED — {name}</span>"
            f"<span>{violations_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Persons Detected", summary["persons"])
    m2.metric("Compliant PPE Items", summary["compliant_count"])
    m3.metric("Violations", summary["violation_count"])
    m4.metric("Total Detections", summary["total_detections"])

    if side_by_side:
        c1, c2 = st.columns(2)
        c1.image(image_rgb, caption="Original Frame", width="stretch")
        c2.image(annotated, caption="PPE Detection Output", width="stretch")
    else:
        st.image(annotated, caption=f"PPE Detection Output — {name}", width="stretch")

    dl_col, _ = st.columns([1, 3])
    dl_col.download_button(
        label="⬇️ Download Annotated PNG",
        data=image_to_png_bytes(annotated),
        file_name=f"annotated_{Path(name).stem}.png",
        mime="image/png",
        key=f"dl_{name}_{len(detections)}_{image_rgb.shape[0]}",
    )

    df = summary["dataframe"]
    if not df.empty:
        with st.expander(f"Detection Details ({len(df)} objects)", expanded=True):
            t_col, chart_col = st.columns([1.3, 1])
            t_col.dataframe(df, width="stretch", hide_index=True)
            fig = px.bar(
                df.groupby(["Class", "Category"], as_index=False).size(),
                x="Class",
                y="size",
                color="Category",
                color_discrete_map={
                    "Compliant": "#22c55e",
                    "Violation": "#ef4444",
                    "Anchor": "#38bdf8",
                },
                labels={"size": "Count"},
                title="Detections by Class",
            )
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
            chart_col.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No objects exceeded the current confidence threshold.")

    return summary, annotated


def render_image_tab(cfg: dict, detector_fn, err_msg: str | None):
    """Render the Image Inspection tab (uploads, dataset samples, batch summary)."""
    st.subheader("Image & Batch PPE Inspection")

    sample_paths = discover_sample_images()
    source_choices = ["Upload Image(s)", "Synthetic Sample Scene"]
    if sample_paths:
        source_choices.insert(1, "Workspace Test / Output Images")

    input_mode = st.radio("Input Source", options=source_choices, horizontal=True)
    side_by_side = st.checkbox("Show Original vs. Annotated Side-by-Side", value=True)

    images_to_run: list[tuple[str, np.ndarray]] = []
    if input_mode == "Upload Image(s)":
        uploads = st.file_uploader(
            "Upload one or more CCTV / worksite images (all image formats supported)",
            type=None,
            accept_multiple_files=True,
            key="image_uploader_any",
        )
        for up in uploads or []:
            try:
                images_to_run.append((up.name, decode_image_bytes(up.getvalue(), up.name)))
            except ValueError as exc:
                st.error(str(exc))
    elif input_mode == "Workspace Test / Output Images":
        rel_map = {str(p.relative_to(PROJECT_ROOT)): p for p in sample_paths}
        picked = st.multiselect(
            "Select workspace images",
            options=list(rel_map.keys()),
            default=list(rel_map.keys())[:1],
        )
        for rel in picked:
            try:
                raw = rel_map[rel].read_bytes()
                images_to_run.append((Path(rel).name, decode_image_bytes(raw, rel)))
            except ValueError as exc:
                st.error(str(exc))
    else:
        images_to_run.append(("synthetic_worksite.png", create_synthetic_site_image()))

    if err_msg:
        st.error(err_msg)
        return

    if not images_to_run:
        st.info("Upload an image above or switch to **Synthetic Sample Scene** to preview PPE detection.")
        return

    batch_rows = []
    for img_name, img_rgb in images_to_run:
        detections = detector_fn(img_rgb)
        summary, _ = render_detection_result(img_name, img_rgb, detections, cfg, side_by_side=side_by_side)
        batch_rows.append(
            {
                "Image": img_name,
                "Status": "Compliant" if summary["is_compliant"] else "Violation",
                "Persons": summary["persons"],
                "Compliant Items": summary["compliant_count"],
                "Violations": summary["violation_count"],
                "Violation Types": ", ".join(summary["violation_types"]) or "None",
            }
        )
        st.divider()

    if len(batch_rows) > 1:
        st.subheader("Batch Compliance Summary")
        batch_df = pd.DataFrame(batch_rows)
        total_imgs = len(batch_df)
        violating_imgs = int((batch_df["Violations"] > 0).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Images Processed", total_imgs)
        c2.metric("Images With Violations", violating_imgs)
        c3.metric("Compliance Rate", f"{((total_imgs - violating_imgs) / total_imgs) * 100:.1f}%")
        st.dataframe(batch_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download Batch Report (CSV)",
            data=batch_df.to_csv(index=False).encode("utf-8"),
            file_name="ppe_batch_report.csv",
            mime="text/csv",
        )


def render_video_tab(cfg: dict, detector_fn, err_msg: str | None):
    """Render the Video / CCTV Footage analysis tab."""
    st.subheader("Video & CCTV Stream Analysis")

    if err_msg:
        st.error(err_msg)
        return

    v_mode = st.radio("Video Source", ["Upload Video File", "Local Path / RTSP URL"], horizontal=True)
    video_source: str | None = None
    tmp_video_file = None

    if v_mode == "Upload Video File":
        uploaded_vid = st.file_uploader(
            "Upload CCTV / site video (all video formats supported)",
            type=None,
            key="video_uploader_any",
        )
        if uploaded_vid is not None:
            suffix = Path(uploaded_vid.name).suffix or ".mp4"
            tmp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_video_file.write(uploaded_vid.getvalue())
            tmp_video_file.close()
            video_source = tmp_video_file.name
    else:
        stream_input = st.text_input("Path or RTSP URL", placeholder="e.g. samples/cctv_gate.mp4 or rtsp://...")
        if stream_input.strip():
            video_source = stream_input.strip()

    col1, col2 = st.columns(2)
    frame_stride = col1.slider("Process every Nth frame", 1, 30, 5)
    max_frames = col2.slider("Max sampled frames", 10, 500, 100, step=10)

    if st.button("▶️ Run Video Analysis", type="primary", disabled=video_source is None):
        progress_bar = st.progress(0, text="Initializing video pipeline...")
        preview_slot = st.empty()
        kpi_slot = st.empty()
        violation_snapshots: list[tuple[int, np.ndarray, str]] = []

        def _on_frame(done: int, total: int, annotated_rgb: np.ndarray, summary: dict):
            pct = min(1.0, done / max(1, total))
            progress_bar.progress(pct, text=f"Analyzed {done}/{total} sampled frames")
            preview_slot.image(annotated_rgb, caption=f"Sample #{done}", width="stretch")
            with kpi_slot.container():
                k1, k2, k3 = st.columns(3)
                k1.metric("Persons (frame)", summary["persons"])
                k2.metric("Compliant (frame)", summary["compliant_count"])
                k3.metric("Violations (frame)", summary["violation_count"])
            if not summary["is_compliant"] and len(violation_snapshots) < 6:
                violation_snapshots.append(
                    (done, annotated_rgb, ", ".join(summary["violation_types"]))
                )

        out_dir = PROJECT_ROOT / "outputs" / "visualizations"
        out_path = out_dir / "streamlit_annotated_video.mp4"
        try:
            saved_path, log_df = process_video(
                video_path=video_source,
                output_path=out_path,
                detector_fn=detector_fn,
                frame_stride=frame_stride,
                max_frames=max_frames,
                allowed_classes=cfg["allowed_classes"],
                show_labels=cfg["show_labels"],
                show_conf=cfg["show_conf"],
                line_width=cfg["line_width"],
                progress_cb=_on_frame,
            )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
            return
        finally:
            if tmp_video_file is not None:
                Path(tmp_video_file.name).unlink(missing_ok=True)

        progress_bar.progress(1.0, text="Video analysis complete!")
        if log_df.empty:
            st.warning("No frames were decoded from the video source.")
            return

        st.success(f"Processed {len(log_df)} sampled frames. Saved annotated video to `{saved_path.relative_to(PROJECT_ROOT)}`.")

        fig = px.line(
            log_df,
            x="Time (s)",
            y=["Persons", "Compliant", "Violations"],
            title="PPE Compliance Timeline",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        if violation_snapshots:
            st.markdown("#### Key Violation Frames")
            cols = st.columns(min(3, len(violation_snapshots)))
            for idx, (sample_num, snap_img, v_types) in enumerate(violation_snapshots):
                cols[idx % len(cols)].image(
                    snap_img,
                    caption=f"Sample #{sample_num}: {v_types}",
                    width="stretch",
                )

        c_vid, c_csv = st.columns(2)
        c_vid.download_button(
            "⬇️ Download Annotated Video (.mp4)",
            data=saved_path.read_bytes(),
            file_name=saved_path.name,
            mime="video/mp4",
        )
        c_csv.download_button(
            "⬇️ Download Frame Violation Log (.csv)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="video_ppe_log.csv",
            mime="text/csv",
        )
        st.dataframe(log_df, width="stretch", hide_index=True)


def render_camera_tab(cfg: dict, detector_fn, err_msg: str | None):
    """Render Live Camera snapshot capture and PPE compliance check."""
    st.subheader("Live Webcam Compliance Check")
    if err_msg:
        st.error(err_msg)
        return

    snap = st.camera_input("Capture a camera frame to inspect PPE compliance")
    if snap is None:
        st.info("Allow browser camera access and click **Take Photo** to run instant PPE detection.")
        return

    img_rgb = decode_image_bytes(snap.getvalue(), "webcam_snapshot.png")
    detections = detector_fn(img_rgb)
    render_detection_result("webcam_snapshot.png", img_rgb, detections, cfg, side_by_side=True)


def render_dashboard_tab():
    """Render Dataset Statistics and Training Run Metrics dashboard."""
    st.subheader("Dataset & Training Overview")

    ds_cfg = load_dataset_config()
    names = ds_cfg.get("names", {})
    with st.expander("📄 Dataset Configuration (`dataset/ppe_dataset.yaml`)", expanded=False):
        st.json(ds_cfg)

    image_counts, class_df = scan_dataset_stats()
    total_ds_imgs = sum(image_counts.values())

    st.markdown("#### Dataset Splits")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Train Images", image_counts.get("train", 0))
    s2.metric("Val Images", image_counts.get("val", 0))
    s3.metric("Test Images", image_counts.get("test", 0))
    s4.metric("Configured Classes", ds_cfg.get("nc", len(names)))

    if total_ds_imgs > 0 and not class_df.empty:
        fig = px.bar(
            class_df,
            x="Class",
            y="Instances",
            color="Split",
            barmode="group",
            title="Ground-Truth Label Distribution Across Splits",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Dataset images have not been prepared locally yet. Run:\n\n"
            "```bash\npython scripts/prepare_dataset.py --val_ratio 0.15 --test_ratio 0.05\n```"
        )

    st.divider()
    exported_plots = discover_exported_plots()
    if exported_plots:
        st.markdown("#### Exported Training & Evaluation Plots (`plots/`)")
        exp_cols = st.columns(2)
        for idx, plot_path in enumerate(exported_plots):
            exp_cols[idx % 2].image(str(plot_path), caption=plot_path.name, width="stretch")
        st.divider()

    st.markdown("#### Training Runs (`runs/train/`)")
    runs = load_training_runs()
    if not runs:
        if not exported_plots:
            st.info(
                "No local training runs found in `runs/train/`. Train or evaluate using:\n\n"
                "```bash\n"
                "python scripts/train.py --model yolov8n --epochs 50 --batch 16\n"
                "python scripts/evaluate.py --weights runs/train/ppe_yolov8n/weights/best.pt\n"
                "```"
            )
        else:
            st.caption("Metrics above were exported from your training run into `plots/` and `weights/best.onnx`.")
        return

    run_map = {r["name"]: r for r in runs}
    selected_run_name = st.selectbox("Select Training Run", options=list(run_map.keys()))
    run = run_map[selected_run_name]

    if run["weights"]:
        st.caption("Available weights: " + ", ".join(f"`{w.name}`" for w in run["weights"]))

    df = run["metrics_df"]
    if df is not None and not df.empty and "epoch" in df.columns:
        metric_cols = [
            c
            for c in (
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(B)",
                "metrics/recall(B)",
            )
            if c in df.columns
        ]
        loss_cols = [c for c in ("train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss") if c in df.columns]

        c_left, c_right = st.columns(2)
        if metric_cols:
            c_left.plotly_chart(
                px.line(df, x="epoch", y=metric_cols, title="Validation Metrics"),
                use_container_width=True,
            )
        if loss_cols:
            c_right.plotly_chart(
                px.line(df, x="epoch", y=loss_cols, title="Training & Validation Losses"),
                use_container_width=True,
            )

    if run["plots"]:
        st.markdown("##### Ultralytics Training Artifacts")
        plot_cols = st.columns(2)
        for idx, plot_path in enumerate(run["plots"]):
            plot_cols[idx % 2].image(str(plot_path), caption=plot_path.name, width="stretch")


def main():
    cfg = render_sidebar()

    st.title("🦺 PPE Compliance Violation Detection")
    st.caption(
        "YOLOv8 real-time monitoring for hardhats, safety vests, and gloves "
        "(9 classes: compliant gear, improper wear, and missing PPE)."
    )

    if cfg["demo_mode"]:
        st.warning(
            "**Demo Mode Active** — Showing synthetic placeholder detections for UI exploration. "
            "Disable **Demo Mode** in the sidebar and load trained `.pt` weights for real YOLOv8 predictions."
        )

    detector_fn, err_msg = make_detector(cfg)

    tab_img, tab_vid, tab_cam, tab_dash = st.tabs(
        [
            "📸 Image Inspection",
            "🎥 Video / CCTV Analysis",
            "📷 Live Camera",
            "📊 Dataset & Training",
        ]
    )

    with tab_img:
        render_image_tab(cfg, detector_fn, err_msg)
    with tab_vid:
        render_video_tab(cfg, detector_fn, err_msg)
    with tab_cam:
        render_camera_tab(cfg, detector_fn, err_msg)
    with tab_dash:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
