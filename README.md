# 🦺 PPE Compliance Violation Detection — YOLOv8

Detect **Personal Protective Equipment (PPE) violations** in CCTV footage using YOLOv8.

> **Dataset:** [Simuletic CCTV-PPE Compliance Detection Dataset](https://github.com/Simuletic/cctv-ppe-compliance-detection-dataset)  
> **Model:** YOLOv8 (Ultralytics) — trained from COCO pretrained weights  
> **Classes:** 9 (person, hardhat, hardhat_wrong, no_hardhat, vest, vest_wrong, no_vest, gloves, no_gloves)

---

## 📁 Project Structure

```
gril_3_assessment/
│
├── dataset/
│   ├── ppe_dataset.yaml          ← YOLO dataset config (paths + class names)
│   ├── raw/                      ← cloned dataset goes here
│   ├── images/
│   │   ├── train/                ← 80% of images
│   │   ├── val/                  ← 15% of images
│   │   └── test/                 ← 5%  of images
│   └── labels/
│       ├── train/                ← YOLO .txt labels (mirror of images/)
│       ├── val/
│       └── test/
│
├── models/
│   ├── weights/                  ← place pretrained / custom .pt files here
│   └── configs/
│       └── yolov8_ppe.yaml       ← custom YOLOv8 architecture config
│
├── notebooks/
│   └── ppe_detection_training.ipynb  ← full end-to-end GPU notebook ✅
│
├── scripts/
│   ├── prepare_dataset.py        ← clone + split dataset
│   ├── train.py                  ← training script
│   ├── detect.py                 ← inference + violation summary
│   └── evaluate.py               ← mAP / precision / recall on test set
│
├── utils/
│   └── visualize.py              ← bounding-box drawing helpers
│
├── runs/
│   ├── train/                    ← Ultralytics training runs (auto-generated)
│   └── detect/                   ← Ultralytics detection runs
│
├── outputs/
│   ├── predictions/              ← saved inference images + labels
│   └── visualizations/           ← custom annotated outputs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# Install PyTorch with CUDA (adjust CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare the dataset
```bash
python scripts/prepare_dataset.py --val_ratio 0.15 --test_ratio 0.05
```

### 3. Train
```bash
python scripts/train.py --model yolov8n --epochs 50 --batch 16
```

### 4. Evaluate
```bash
python scripts/evaluate.py --weights runs/train/ppe_yolov8n/weights/best.pt
```

### 5. Detect violations on new images
```bash
python scripts/detect.py --source path/to/images --conf 0.4
```

---

## 🧠 Model Details

| Variant | Params | Speed (ms) | mAP (COCO) | Recommended for |
|---------|--------|-----------|-----------|----------------|
| YOLOv8n | 3.2M | ~1.5 | 37.3 | Quick experiments, low GPU memory |
| YOLOv8s | 11.2M | ~2.3 | 44.9 | Balanced speed / accuracy |
| YOLOv8m | 25.9M | ~5.1 | 50.2 | Assignment submission model |
| YOLOv8l | 43.7M | ~8.1 | 52.9 | High-accuracy production |
| YOLOv8x | 68.2M | ~12.6 | 53.9 | Best accuracy, high VRAM needed |

> **Recommended:** `yolov8n` for fast iteration; `yolov8s` or `yolov8m` for final submission.

---

## 🏷️ Detection Classes

| ID | Class | Type |
|----|-------|------|
| 0 | `person` | Anchor |
| 1 | `hardhat` | ✅ Compliant |
| 2 | `hardhat_wrong` | ⚠️ Violation |
| 3 | `no_hardhat` | ❌ Violation |
| 4 | `vest` | ✅ Compliant |
| 5 | `vest_wrong` | ⚠️ Violation |
| 6 | `no_vest` | ❌ Violation |
| 7 | `gloves` | ✅ Compliant |
| 8 | `no_gloves` | ❌ Violation |

---

## 📊 Training Hyperparameters (defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | stable convergence |
| Learning rate | 1e-3 → 1e-5 | cosine decay |
| Warmup epochs | 3 | |
| Augmentation | Mosaic + MixUp + CopyPaste | heavy augmentation for small dataset |
| Image size | 640 × 640 | standard YOLO input |
| Early stopping | 20 epochs | patience |

---

## 📓 Jupyter Notebook (GPU Training)
Open `notebooks/ppe_detection_training.ipynb` for a fully self-contained walkthrough:
1. Environment check
2. Dataset download & split
3. Class distribution analysis
4. Training with live metrics
5. Evaluation (mAP, precision, recall)
6. Visualized predictions with violation summary
7. ONNX export

---

## 🔧 Tips for GPU Notebook (Colab / Kaggle)
- Select **GPU runtime** (Runtime → Change runtime type → T4/A100)
- Run `!nvidia-smi` to confirm GPU
- Reduce `BATCH_SIZE` to 8 if you get OOM errors
- Use `yolov8n` first to validate the pipeline, then switch to `yolov8m`
