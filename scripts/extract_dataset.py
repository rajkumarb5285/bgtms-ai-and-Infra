"""
BGTMS - Dataset Extraction & Standardizer (M1-06)
Unzips raw datasets, parses diverse annotation formats (YOLO, Pascal VOC, COCO),
and organizes them into a unified format in data/processed/.
"""

import os
import zipfile
import shutil
import argparse
from pathlib import Path


import os
import zipfile
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# Harmonized BGTMS mapping
CLASS_MAP = {
    "car": 0,
    "automobile": 0,
    "suv": 0,
    "sedan": 0,
    "motorcycle": 1,
    "two_wheelers": 1,
    "two_wheeler": 1,
    "motorbike": 1,
    "scooter": 1,
    "bike": 1,
    "auto": 2,
    "autorickshaw": 2,
    "auto_rickshaw": 2,
    "three_wheeler": 2,
    "bus": 3,
    "truck": 4,
    "lorry": 4,
    "tempo": 4,
    "bicycle": 5,
    "cycle": 5,
    "pedestrian": 6,
    "person": 6,
    "ambulance": 7,
    "fire_truck": 7,
    "police": 7,
    "emergency_vehicle": 7
}


def extract_zip(zip_path, extract_to):
    print(f"[*] Extracting {zip_path} -> {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[+] Extraction complete for {os.path.basename(zip_path)}")


def convert_voc_xml_to_yolo(xml_file, output_label_file, class_counts):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        size = root.find("size")
        if size is None:
            return 0
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        if width == 0 or height == 0:
            return 0

        yolo_lines = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip().lower()
            if raw_name not in CLASS_MAP:
                continue
            class_id = CLASS_MAP[raw_name]
            class_counts[raw_name] = class_counts.get(raw_name, 0) + 1

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Clamp coordinates
            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))
            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert to YOLO center x, center y, w, h normalized
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            bbox_w = (xmax - xmin) / width
            bbox_h = (ymax - ymin) / height

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

        with open(output_label_file, "w", encoding="utf-8") as wf:
            wf.writelines(yolo_lines)
            
        return len(yolo_lines)
    except Exception as e:
        print(f"[!] Error converting {xml_file}: {e}")
        return 0


def process_extracted_dataset(extracted_dir, processed_dir):
    proc_img_dir = Path(processed_dir) / "images"
    proc_lbl_dir = Path(processed_dir) / "labels"
    proc_img_dir.mkdir(parents=True, exist_ok=True)
    proc_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Standardizing annotations from {extracted_dir} -> {processed_dir} ...")
    
    class_counts = {}
    total_images_processed = 0
    total_boxes = 0

    for root, _, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = Path(root) / f
                stem = img_path.stem
                xml_path = img_path.with_suffix(".xml")

                if not xml_path.exists():
                    continue

                dest_img = proc_img_dir / f"{stem}.jpg"
                dest_lbl = proc_lbl_dir / f"{stem}.txt"

                # Convert XML to YOLO
                box_count = convert_voc_xml_to_yolo(xml_path, dest_lbl, class_counts)
                if box_count > 0:
                    # Copy and resize image to standardized max 1280 for fast training & disk efficiency
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                        img.save(dest_img, "JPEG", quality=90)
                    total_images_processed += 1
                    total_boxes += box_count

    print("\n[+] Standardized Dataset Summary:")
    print(f"    - Processed Images: {total_images_processed}")
    print(f"    - Total Bounding Boxes: {total_boxes}")
    print("    - Class Breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"        * {cls_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Extract and standardize raw traffic datasets")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw zip archives")
    parser.add_argument("--extract-dir", default="data/extracted", help="Extraction directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed standard directory")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    extract_path = Path(args.extract_dir)

    zip_files = list(raw_path.glob("*.zip"))
    for zf in zip_files:
        target_sub = extract_path / zf.stem
        if not target_sub.exists():
            extract_zip(str(zf), str(target_sub))

    process_extracted_dataset(extract_path, args.processed_dir)


if __name__ == "__main__":
    main()

