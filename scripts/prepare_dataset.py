"""
prepare_dataset.py
------------------
Clones / downloads the Simuletic CCTV-PPE dataset and splits it into
train / val / test sets expected by YOLOv8.

Usage (from project root):
    python scripts/prepare_dataset.py --val_ratio 0.15 --test_ratio 0.05
"""

import os
import shutil
import random
import argparse
from pathlib import Path

DATASET_REPO = "https://github.com/Simuletic/cctv-ppe-compliance-detection-dataset.git"
RAW_DIR      = Path("dataset/raw")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}


def clone_dataset(raw_dir: Path):
    if (raw_dir / ".git").exists():
        print("[INFO] Dataset already cloned – skipping git clone.")
        return
    print(f"[INFO] Cloning dataset from {DATASET_REPO} …")
    os.system(f"git clone {DATASET_REPO} {raw_dir}")


def collect_pairs(raw_dir: Path):
    """Return list of (image_path, label_path) pairs."""
    pairs = []
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def copy_split(pairs, split_name: str, base: Path):
    img_dst = base / "images" / split_name
    lbl_dst = base / "labels" / split_name
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_dst / img.name)
        shutil.copy2(lbl, lbl_dst / lbl.name)
    print(f"[INFO] {split_name:5s}: {len(pairs):5d} images")


def main(val_ratio=0.15, test_ratio=0.05, seed=42):
    base = Path("dataset")
    clone_dataset(RAW_DIR)

    pairs = collect_pairs(RAW_DIR)
    if not pairs:
        raise RuntimeError("No image–label pairs found in dataset/raw/. "
                           "Check the clone or folder structure.")

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_test  = int(n_total * test_ratio)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f"[INFO] Total images found: {n_total}")
    copy_split(train_pairs, "train", base)
    copy_split(val_pairs,   "val",   base)
    copy_split(test_pairs,  "test",  base)
    print("[DONE] Dataset prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args.val_ratio, args.test_ratio, args.seed)
