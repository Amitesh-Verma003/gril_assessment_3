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
import zipfile
import argparse
import urllib.request
from pathlib import Path

DATASET_REPO = "https://github.com/Simuletic/cctv-ppe-compliance-detection-dataset.git"
CONSTRUCTION_PPE_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
# Map Ultralytics Construction-PPE IDs -> our 9-class PPE schema
CONSTRUCTION_PPE_MAP = {
    0: 1,   # helmet     -> hardhat
    1: 7,   # gloves     -> gloves
    2: 4,   # vest       -> vest
    5: 0,   # Person     -> person
    6: 3,   # no_helmet  -> no_hardhat
    8: 8,   # no_gloves  -> no_gloves
    11: 6,  # no_vest    -> no_vest
}
RAW_DIR      = Path("dataset/raw")
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clone_dataset(raw_dir: Path):
    if (raw_dir / ".git").exists():
        print("[INFO] Dataset already cloned – skipping git clone.")
        return
    print(f"[INFO] Cloning dataset from {DATASET_REPO} …")
    os.system(f"git clone {DATASET_REPO} {raw_dir}")


def collect_pairs(raw_dir: Path):
    """Return list of (image_path, label_path) pairs from nested train/val."""
    pairs = []
    raw_img_dir = raw_dir / "images"
    raw_lbl_dir = raw_dir / "labels"
    
    for root, _, files in os.walk(raw_img_dir):
        for file in files:
            img_path = Path(root) / file
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            # Relative path from raw_img_dir
            rel_path = img_path.relative_to(raw_img_dir)
            label_path = raw_lbl_dir / rel_path.with_suffix(".txt")
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


def merge_construction_ppe(base: Path):
    """Download Ultralytics Construction-PPE (~1,000 images) and merge into train/val."""
    extra_dir = base / "extra_construction_ppe"
    zip_path = extra_dir / "construction-ppe.zip"
    extra_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"[INFO] Downloading Ultralytics Construction-PPE dataset from {CONSTRUCTION_PPE_ZIP} ...")
        urllib.request.urlretrieve(CONSTRUCTION_PPE_ZIP, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extra_dir)

    # Locate extracted root containing images/ and labels/
    src_root = extra_dir / "construction-ppe"
    if not (src_root / "images").exists():
        src_root = extra_dir

    added = {"train": 0, "val": 0}
    for split in ("train", "val"):
        img_src_dir = src_root / "images" / split
        lbl_src_dir = src_root / "labels" / split
        if not img_src_dir.exists() or not lbl_src_dir.exists():
            continue
        img_dst = base / "images" / split
        lbl_dst = base / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_src_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            remapped_lines = []
            for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if cid in CONSTRUCTION_PPE_MAP:
                    parts[0] = str(CONSTRUCTION_PPE_MAP[cid])
                    remapped_lines.append(" ".join(parts))

            if not remapped_lines:
                continue
            prefix_name = f"cppe_{img_path.name}"
            shutil.copy2(img_path, img_dst / prefix_name)
            (lbl_dst / f"cppe_{img_path.stem}.txt").write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
            added[split] += 1

    print(f"[INFO] Merged Construction-PPE -> train: +{added['train']}, val: +{added['val']} images")


def main(val_ratio=0.15, test_ratio=0.05, seed=42, include_construction_ppe=False):
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

    print(f"[INFO] Total Simuletic images found: {n_total}")
    copy_split(train_pairs, "train", base)
    copy_split(val_pairs,   "val",   base)
    copy_split(test_pairs,  "test",  base)
    if include_construction_ppe:
        merge_construction_ppe(base)
    print("[DONE] Dataset prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--include_construction_ppe", action="store_true",
                        help="Also download & merge Ultralytics Construction-PPE (~1,000 images)")
    args = parser.parse_args()
    main(args.val_ratio, args.test_ratio, args.seed, args.include_construction_ppe)
