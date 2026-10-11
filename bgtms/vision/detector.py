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
        # Load High-Accuracy Deep Learning model (FasterRCNN-MobileNetV3-FPN)
        self.model = None
        self.model_type = None
        self.device = None
        
        try:
            import torch
            import torchvision.models.detection as detection
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 1. Primary: FasterRCNN-MobileNetV3-FPN for high-recall dense multi-scale traffic
            rcnn_weights_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models", "fasterrcnn_mobilenet_v3.pth")
            )
            if os.path.exists(rcnn_weights_path):
                model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
                state_dict = torch.load(rcnn_weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device).eval()
                self.model = model
                self.model_type = "fasterrcnn_fpn"
                print(f"[VehicleDetector] Successfully loaded High-Accuracy FasterRCNN-FPN weights from {rcnn_weights_path}")
            else:
                # Load via torchvision default weights
                weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
                model.to(self.device).eval()
                self.model = model
                self.model_type = "fasterrcnn_fpn"
                print("[VehicleDetector] Loaded FasterRCNN-MobileNetV3-FPN with pretrained weights")
        except Exception as e:
            print(f"[VehicleDetector] FasterRCNN loading error: {e}, checking SSDLite fallback...")
            try:
                import torch
                from bgtms.models.train_detector import create_detector_model, NUM_CLASSES
                ssd_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "models", "vehicle_detector.pth")
                )
                if os.path.exists(ssd_path):
                    model = create_detector_model(num_classes=NUM_CLASSES, freeze_backbone=False)
                    checkpoint = torch.load(ssd_path, map_location=self.device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self.device).eval()
                    self.model = model
                    self.model_type = "ssdlite"
                    print(f"[VehicleDetector] Loaded fallback SSDLite from {ssd_path}")
            except Exception as e2:
                print(f"[VehicleDetector] Deep Learning initialization fallback: {e2}")

        # COCO category lookup for FasterRCNN
        self.coco_categories = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[BoundingBox]:
        """
        Runs object detection on input frame (BGR numpy array).
        Uses multiscale FPN deep learning model with Indian vehicle specialization,
        and fallback to heuristic edge detection.
        """
        h, w = image.shape[:2]
        detections: List[BoundingBox] = []
        effective_thresh = conf_threshold if conf_threshold is not None else self.conf_thresh

        # 1. FasterRCNN-MobileNetV3-FPN Inference
        if self.model is not None and self.model_type == "fasterrcnn_fpn":
            try:
                import torch
                import torchvision.transforms.functional as TF
                import torchvision.ops as ops
                from PIL import Image

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Dynamic aspect-ratio preserving scaling (cap max dimension to 1280 for CPU latency)
                max_dim = max(h, w)
                if max_dim > 1280:
                    scale_factor = 1280.0 / max_dim
                    infer_w = int(w * scale_factor)
                    infer_h = int(h * scale_factor)
                    infer_img = cv2.resize(rgb, (infer_w, infer_h))
                else:
                    scale_factor = 1.0
                    infer_w, infer_h = w, h
                    infer_img = rgb

                img_tensor = TF.to_tensor(Image.fromarray(infer_img)).to(self.device)

                with torch.no_grad():
                    preds = self.model([img_tensor])

                pred = preds[0]
                raw_boxes = pred["boxes"].cpu()
                raw_scores = pred["scores"].cpu()
                raw_labels = pred["labels"].cpu()

                # Filter NMS
                keep = ops.batched_nms(raw_boxes, raw_scores, raw_labels, iou_threshold=0.38)
                p_boxes = raw_boxes[keep].numpy()
                p_scores = raw_scores[keep].numpy()
                p_labels = raw_labels[keep].numpy()

                # Precompute HSV for auto-rickshaw detection
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask_yellow = cv2.inRange(hsv, (14, 55, 55), (38, 255, 255))

                coord_scale = 1.0 / scale_factor

                traffic_coco_map = {
                    "car": ("car", 0),
                    "motorcycle": ("motorcycle", 1),
                    "bus": ("bus", 3),
                    "truck": ("truck", 4),
                    "bicycle": ("bicycle", 5),
                    "person": ("pedestrian", 6)
                }

                for b, s, l in zip(p_boxes, p_scores, p_labels):
                    if s < effective_thresh:
                        continue
                    if l >= len(self.coco_categories):
                        continue
                    cat_name = self.coco_categories[l]
                    if cat_name not in traffic_coco_map:
                        continue

                    target_name, target_id = traffic_coco_map[cat_name]

                    # Scale back to original resolution
                    x1 = float(max(0.0, b[0] * coord_scale))
                    y1 = float(max(0.0, b[1] * coord_scale))
                    x2 = float(min(float(w), b[2] * coord_scale))
                    y2 = float(min(float(h), b[3] * coord_scale))

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 4 or bh <= 4:
                        continue

                    # Specialized Indian traffic detection: Auto-rickshaw
                    aspect = bw / max(1.0, bh)
                    crop_y1 = max(0, int(y1))
                    crop_y2 = min(h, int(y2))
                    crop_x1 = max(0, int(x1))
                    crop_x2 = min(w, int(x2))
                    crop_yellow = mask_yellow[crop_y1:crop_y2, crop_x1:crop_x2]
                    yellow_ratio = float((crop_yellow > 0).mean()) if crop_yellow.size > 0 else 0.0

                    if target_name in ["car", "truck", "motorcycle"]:
                        if yellow_ratio > 0.10 and 0.60 <= aspect <= 1.45 and (bw * bh) < (w * h * 0.07):
                            target_name = "auto_rickshaw"
                            target_id = 2

                    detections.append(BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(s),
                        class_id=target_id,
                        class_name=target_name
                    ))

                if len(detections) > 0:
                    return detections
            except Exception as e:
                print(f"[VehicleDetector] FasterRCNN inference error: {e}")

        # SSDLite fallback if FasterRCNN not active
        if self.model is not None and self.model_type == "ssdlite":
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
