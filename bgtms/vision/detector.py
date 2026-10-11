"""
BGTMS - Vehicle Detection Engine (M1-10 & M1-11)
Detects cars, buses, trucks, motorcycles, autos, bicycles, pedestrians, and emergency vehicles.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from bgtms.core.types import BoundingBox
from bgtms.core.config import get_classes, get_settings


class VehicleDetector:
    """
    Traffic Object Detector for Bengaluru's heterogeneous vehicle stream.
    """

    def __init__(self, confidence_thresh: float = 0.35, iou_thresh: float = 0.45):
        self.conf_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        
        self.classes_cfg = get_classes()
        self.classes = {int(k): v["name"] for k, v in self.classes_cfg.get("classes", {}).items()}
        self.pcu_map = {int(k): v["pcu"] for k, v in self.classes_cfg.get("classes", {}).items()}

        # Color mapping for visualization
        self.colors = {
            0: (246, 130, 59),   # car: blue
            1: (129, 185, 16),   # motorcycle: green
            2: (11, 158, 245),   # auto_rickshaw: amber/orange
            3: (246, 92, 139),   # bus: purple
            4: (128, 114, 107),  # truck: gray
            5: (212, 182, 6),    # bicycle: cyan
            6: (153, 72, 236),   # pedestrian: pink
            7: (68, 68, 239)     # emergency: bright red
        }

        # Load trained PyTorch Deep Learning model if available
        self.model = None
        self.device = None
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "vehicle_detector.pth"))
        if os.path.exists(model_path):
            try:
                import torch
                from bgtms.models.train_detector import create_detector_model, NUM_CLASSES
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = create_detector_model(num_classes=NUM_CLASSES, freeze_backbone=False)
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.model = model
                print(f"[VehicleDetector] Successfully loaded Deep Learning weights from {model_path}")
            except Exception as e:
                print(f"[VehicleDetector] Deep Learning checkpoint fallback: {e}")

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[BoundingBox]:
        """
        Runs object detection on input frame (BGR numpy array).
        Uses fine-tuned deep learning model if loaded, with heuristic fallback.
        """
        h, w = image.shape[:2]
        detections: List[BoundingBox] = []
        effective_thresh = conf_threshold if conf_threshold is not None else self.conf_thresh

        # 1. Deep Learning Inference if model is loaded
        if self.model is not None:
            try:
                import torch
                import torchvision.transforms.functional as TF
                from PIL import Image

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((320, 320))
                img_tensor = TF.to_tensor(pil_img).to(self.device)

                with torch.no_grad():
                    preds = self.model([img_tensor])

                pred = preds[0]
                raw_boxes = pred["boxes"].cpu()
                raw_scores = pred["scores"].cpu()
                raw_labels = pred["labels"].cpu()

                import torchvision.ops as ops
                keep = ops.batched_nms(raw_boxes, raw_scores, raw_labels, iou_threshold=0.40)
                p_boxes = raw_boxes[keep].numpy()
                p_scores = raw_scores[keep].numpy()
                p_labels = raw_labels[keep].numpy()

                scale_x = w / 320.0
                scale_y = h / 320.0

                for b, s, l in zip(p_boxes, p_scores, p_labels):
                    if s >= effective_thresh:
                        cls_id = int(l) - 1
                        if 0 <= cls_id < len(self.classes):
                            detections.append(BoundingBox(
                                x1=float(max(0.0, b[0] * scale_x)),
                                y1=float(max(0.0, b[1] * scale_y)),
                                x2=float(min(float(w), b[2] * scale_x)),
                                y2=float(min(float(h), b[3] * scale_y)),
                                confidence=float(s),
                                class_id=cls_id,
                                class_name=self.classes.get(cls_id, "unknown")
                            ))

                if len(detections) > 0:
                    return detections
            except Exception:
                pass

        # Color analysis / Edge density / Vehicle saliency heuristic for zero-dependency edge execution
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 800 or area > (w * h * 0.4):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter non-vehicle aspect ratios
            aspect = bw / float(bh)
            if 0.3 <= aspect <= 3.5 and y > h * 0.2:
                # Classify based on size and aspect ratio
                cls_id = self._classify_contour(area, aspect, bh, h)
                conf = min(0.98, max(0.40, 0.5 + (area / (w * h * 0.05)) * 0.4))
                boxes.append((x, y, x + bw, y + bh, conf, cls_id))

        # Apply Non-Maximum Suppression (NMS)
        if boxes:
            boxes_np = np.array(boxes)
            nms_boxes = self._apply_nms(boxes_np, self.iou_thresh)
            for b in nms_boxes:
                x1, y1, x2, y2, conf, cls_id = b
                cls_id = int(cls_id)
                detections.append(BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.classes.get(cls_id, "unknown")
                ))

        return detections

    def _classify_contour(self, area: float, aspect: float, bh: float, frame_h: float) -> int:
        """
        Deterministic geometry-based class classifier for Indian vehicle shapes:
        - 2-wheelers: narrow (aspect < 0.7)
        - Autos: compact, tall-boxy (aspect 0.7 - 1.1)
        - Cars: wide (aspect > 1.3)
        - Buses/Trucks: large area (> 4% frame area)
        """
        rel_area = area / (frame_h * frame_h)
        if rel_area > 0.08:
            return 3 if aspect > 1.2 else 4  # Bus or Truck
        elif aspect < 0.75:
            return 1  # Motorcycle / Scooter
        elif 0.75 <= aspect <= 1.2:
            return 2  # Auto-Rickshaw
        elif aspect > 1.2:
            return 0  # Car
        return 0

    def _apply_nms(self, boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0, 6))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return boxes[keep]

    def annotate(self, image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
        """
        Draws professional bounding boxes and labels on image.
        """
        annotated = image.copy()
        for det in detections:
            color = self.colors.get(det.class_id, (0, 255, 0))
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

            # Box outline with rounded aesthetic
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
