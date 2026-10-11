"""
BGTMS - PyTorch Traffic Object Detection Dataset
Loads standardized Bengaluru traffic frames and YOLO annotations.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF


class IndianVehicleDataset(Dataset):
    """
    PyTorch Dataset for Indian Heterogeneous Vehicle Stream.
    Formats targets for SSDLite / Faster R-CNN detectors:
    - boxes: FloatTensor[N, 4] in (x1, y1, x2, y2) absolute pixel coordinates
    - labels: Int64Tensor[N] in [1, 8] (0 = background)
    """

    def __init__(self, split_file: str, img_size: Tuple[int, int] = (320, 320), augment: bool = False):
        self.split_file = Path(split_file)
        self.img_size = img_size
        self.augment = augment

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.image_paths = [Path(line.strip()) for line in f if line.strip()]

        # Filter only existing paths
        self.image_paths = [p for p in self.image_paths if p.exists()]
        print(f"[Dataset] Loaded {len(self.image_paths)} images from {self.split_file.name} (augment={augment})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Find corresponding label file
        label_path = Path("data/processed/labels") / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            xc = float(parts[1]) * orig_w
                            yc = float(parts[2]) * orig_h
                            w = float(parts[3]) * orig_w
                            h = float(parts[4]) * orig_h

                            x1 = max(0.0, xc - w / 2.0)
                            y1 = max(0.0, yc - h / 2.0)
                            x2 = min(float(orig_w), xc + w / 2.0)
                            y2 = min(float(orig_h), yc + h / 2.0)

                            if x2 > x1 + 2.0 and y2 > y1 + 2.0:
                                boxes.append([x1, y1, x2, y2])
                                # Target label: cls_id + 1 (1..8)
                                labels.append(cls_id + 1)
                        except (ValueError, IndexError):
                            continue

        # Resize image to target dimension
        target_w, target_h = self.img_size
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)

        # Scale bounding box coordinates
        scaled_boxes = []
        for b in boxes:
            sx1 = b[0] * scale_x
            sy1 = b[1] * scale_y
            sx2 = b[2] * scale_x
            sy2 = b[3] * scale_y
            scaled_boxes.append([sx1, sy1, sx2, sy2])

        # Data augmentation during training
        if self.augment:
            # 1. Random horizontal flip (50% probability)
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_boxes = []
                for b in scaled_boxes:
                    flipped_x1 = target_w - b[2]
                    flipped_x2 = target_w - b[0]
                    flipped_boxes.append([flipped_x1, b[1], flipped_x2, b[3]])
                scaled_boxes = flipped_boxes

            # 2. Color / Brightness Jitter
            if torch.rand(1).item() > 0.5:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                image = ImageEnhance.Brightness(image).enhance(factor)
            if torch.rand(1).item() > 0.5:
                factor = 0.8 + 0.4 * torch.rand(1).item()
                image = ImageEnhance.Contrast(image).enhance(factor)

        # Convert image to FloatTensor [3, H, W] in [0, 1]
        img_tensor = TF.to_tensor(image)

        # Build target dict
        if len(scaled_boxes) > 0:
            boxes_tensor = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(scaled_boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        return img_tensor, target


def detection_collate_fn(batch):
    """
    Collate function for variable number of objects per image.
    """
    return tuple(zip(*batch))
