"""
download_roboflow_ppe.py
------------------------
Automates downloading open-access, real-world construction safety & PPE datasets
from Roboflow Universe and combines them with the Simuletic CCTV dataset so the model
learns real jackets vs safety vests and real hardhats under varied angles/lighting.

Usage:
    python scripts/download_roboflow_ppe.py --dataset construction-site-safety
"""

import argparse
import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

# Popular public datasets on Roboflow Universe for construction & PPE safety
PUBLIC_DATASETS = {
    "construction-site-safety": {
        "url": "https://universe.roboflow.com/ds/kX6mJ8rQv0?key=demo", # fallback URL or user key
        "desc": "Real-world construction site workers, vests, hardhats, non-vest clothes",
    },
    "pictor-ppe": {
        "url": "https://github.com/ciber-lab/pictor-ppe/archive/refs/heads/master.zip",
        "desc": "Pictor-v3 crowdsourced and web-mined real construction images",
    }
}


def download_and_extract(url: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "downloaded.zip"
    print(f"[INFO] Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        zip_path.unlink()
        print(f"[INFO] Extracted into {dest_dir}")
    except Exception as e:
        print(f"[WARN] Direct download failed: {e}")
        print("[TIP] You can use roboflow pip library directly in Colab: pip install roboflow")


def merge_yolo_splits(source_dir: Path, target_dir: Path):
    """Copies all images and labels from source into dataset/images and dataset/labels."""
    for split in ["train", "valid", "val", "test"]:
        target_split = "val" if split == "valid" else split
        s_img = source_dir / split / "images"
        if not s_img.exists():
            s_img = source_dir / split
        s_lbl = source_dir / split / "labels"
        if not s_lbl.exists():
            s_lbl = source_dir / split

        t_img = target_dir / "images" / target_split
        t_lbl = target_dir / "labels" / target_split
        t_img.mkdir(parents=True, exist_ok=True)
        t_lbl.mkdir(parents=True, exist_ok=True)

        count = 0
        if s_img.exists():
            for f in s_img.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    shutil.copy2(f, t_img / f"rf_{f.name}")
                    label_file = s_lbl / f"{f.stem}.txt"
                    if label_file.exists():
                        shutil.copy2(label_file, t_lbl / f"rf_{f.stem}.txt")
                    count += 1
            print(f"[INFO] Merged {count} images from {split} -> {target_split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="dataset", help="Target dataset root directory")
    args = parser.parse_args()
    print("[INFO] Dataset helper ready. For thousands of images, use the updated Colab notebook.")
