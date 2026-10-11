"""
BGTMS - Leakage-Safe Train/Validation/Test Split Generator (M1-09)
Splits standardized dataset into 70% Train, 15% Validation, and 15% Test sets.
Generates YOLO data YAML config and manifest JSON.
"""

import os
import json
import random
import argparse
from pathlib import Path
import yaml


def create_splits(images_dir, output_dir="data/splits", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    img_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total = len(images)
    print(f"[*] Creating splits for {total} images (Seed={seed}) ...")

    # Shuffle deterministically
    random.shuffle(images)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_set = images[:n_train]
    val_set = images[n_train:n_train + n_val]
    test_set = images[n_train + n_val:]

    # Write text split files with relative paths
    for name, s in [("train", train_set), ("val", val_set), ("test", test_set)]:
        txt_path = out_dir / f"{name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in s:
                f.write(f"{p.as_posix()}\n")
        print(f"[+] {name.capitalize()} split: {len(s)} images ({txt_path})")

    # Generate YOLO dataset.yaml
    yolo_yaml = {
        "path": Path("data").resolve().as_posix(),
        "train": (out_dir / "train.txt").resolve().as_posix(),
        "val": (out_dir / "val.txt").resolve().as_posix(),
        "test": (out_dir / "test.txt").resolve().as_posix(),
        "names": {
            0: "car",
            1: "motorcycle",
            2: "auto_rickshaw",
            3: "bus",
            4: "truck",
            5: "bicycle",
            6: "pedestrian",
            7: "emergency_vehicle"
        }
    }
    yolo_yaml_path = out_dir / "dataset.yaml"
    with open(yolo_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yolo_yaml, f, sort_keys=False)

    # Manifest metadata
    manifest = {
        "total_images": total,
        "seed": seed,
        "splits": {
            "train_count": len(train_set),
            "val_count": len(val_set),
            "test_count": len(test_set)
        },
        "classes": yolo_yaml["names"],
        "dataset_yaml": str(yolo_yaml_path)
    }
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Generated dataset config: {yolo_yaml_path}")
    print(f"[+] Generated split manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--images", default="data/processed/images", help="Processed images directory")
    parser.add_argument("--out", default="data/splits", help="Output directory for manifests")
    args = parser.parse_args()
    create_splits(args.images, args.out)
