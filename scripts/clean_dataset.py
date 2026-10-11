"""
BGTMS - Data Cleaning & Standardization (M1-07)
Removes corrupted images, zero-byte files, detects duplicate images via perceptual/hash deduplication,
and standardizes images and bounding box labels.
"""

import os
import hashlib
import argparse
from pathlib import Path
from PIL import Image
import yaml


def load_classes():
    config_path = Path("configs/classes.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def compute_image_hash(filepath):
    try:
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return None


def clean_dataset(source_dir, output_dir):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_images = output_path / "images"
    output_labels = output_path / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting Data Cleaning on: {source_dir} -> {output_dir}")
    
    seen_hashes = {}
    corrupted_count = 0
    duplicate_count = 0
    valid_count = 0
    missing_label_count = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Find all images
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions:
                # 1. Check file size
                if file_path.stat().st_size == 0:
                    corrupted_count += 1
                    continue

                # 2. Verify image integrity with PIL
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception:
                    corrupted_count += 1
                    continue

                # 3. Check for duplicates via hash
                img_hash = compute_image_hash(file_path)
                if img_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes[img_hash] = file_path

                # 4. Find corresponding label file
                stem = file_path.stem
                label_candidates = [
                    file_path.with_suffix(".txt"),
                    file_path.parent.parent / "labels" / f"{stem}.txt",
                    file_path.parent / f"{stem}.txt"
                ]
                label_file = None
                for cand in label_candidates:
                    if cand.exists():
                        label_file = cand
                        break

                # Copy valid image
                dest_img = output_images / f"{stem}.jpg"
                try:
                    with Image.open(file_path) as img:
                        rgb_img = img.convert("RGB")
                        rgb_img.save(dest_img, "JPEG", quality=95)
                except Exception:
                    corrupted_count += 1
                    continue

                # Copy label if exists, or create empty if background
                dest_label = output_labels / f"{stem}.txt"
                if label_file and label_file.exists():
                    try:
                        with open(label_file, "r", encoding="utf-8") as rf, open(dest_label, "w", encoding="utf-8") as wf:
                            lines = rf.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    wf.write(line)
                    except Exception:
                        pass
                else:
                    dest_label.touch()
                    missing_label_count += 1

                valid_count += 1

    print("\n[+] Data Cleaning Summary:")
    print(f"    - Valid & Processed Images: {valid_count}")
    print(f"    - Corrupted / Unreadable:  {corrupted_count}")
    print(f"    - Duplicates Filtered:     {duplicate_count}")
    print(f"    - Missing Labels (Auto):   {missing_label_count}")
    print(f"    - Cleaned Directory:       {output_dir}")
    return valid_count, corrupted_count, duplicate_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and standardize traffic dataset")
    parser.add_argument("--src", default="data/extracted", help="Extracted data directory")
    parser.add_argument("--out", default="data/processed", help="Clean output directory")
    args = parser.parse_args()
    clean_dataset(args.src, args.out)
