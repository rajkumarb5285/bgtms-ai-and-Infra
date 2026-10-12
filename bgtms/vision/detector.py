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

    def __init__(self, confidence_thresh: float = 0.35, iou_thresh: float = 0.45, enable_sahi: bool = True):
        self.conf_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.enable_sahi = enable_sahi
        
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

        # Load trained Deep Learning model
        self.model = None
        self.model_type = None
        self.device = None
        self.reload_model_weights()

    def reload_model_weights(self):
        """Hot-reloads the latest model weights from disk without restarting service."""
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

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None, enable_sahi: Optional[bool] = None) -> List[BoundingBox]:
        """
        Runs object detection on input frame (BGR numpy array).
        Uses multiscale FPN deep learning model with SAHI (Slicing Aided Hyper Inference)
        for high-density distant traffic queues, with Indian vehicle specialization and fallback heuristics.
        """
        h, w = image.shape[:2]
        detections: List[BoundingBox] = []
        effective_thresh = conf_threshold if conf_threshold is not None else self.conf_thresh
        use_sahi = self.enable_sahi if enable_sahi is None else enable_sahi

        # 1. FasterRCNN-MobileNetV3-FPN Inference with SAHI
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

                coord_scale = 1.0 / scale_factor

                all_raw_boxes = []
                all_raw_scores = []
                all_raw_labels = []

                # Full frame forward pass
                img_tensor = TF.to_tensor(Image.fromarray(infer_img)).to(self.device)
                with torch.no_grad():
                    preds = self.model([img_tensor])

                pred = preds[0]
                fb = pred["boxes"].cpu().numpy()
                fs = pred["scores"].cpu().numpy()
                fl = pred["labels"].cpu().numpy()
                for b, s, l in zip(fb, fs, fl):
                    all_raw_boxes.append([b[0] * coord_scale, b[1] * coord_scale, b[2] * coord_scale, b[3] * coord_scale])
                    all_raw_scores.append(float(s))
                    all_raw_labels.append(int(l))

                # Multi-Scale SAHI Sliding Window Slices for small & distant vehicles (Optimized high-throughput mode)
                if use_sahi and w >= 360 and h >= 180:
                    # Dynamically size tiles to cover frame in at most 6 strategic high-resolution slices
                    # Keeps CPU latency fast (1.5-2.5s) while fully resolving distant queues
                    tile_w = max(260, min(w, int(w * 0.55)))
                    tile_h = max(220, min(h, int(h * 0.60)))
                    step_x = max(120, int(tile_w * 0.75))
                    step_y = max(100, int(tile_h * 0.75))

                    slices = []
                    for y0 in range(0, h, step_y):
                        y1 = min(h, y0 + tile_h)
                        if y1 - y0 < 60:
                            continue
                        for x0 in range(0, w, step_x):
                            x1 = min(w, x0 + tile_w)
                            if x1 - x0 < 60:
                                continue
                            slices.append((x0, y0, x1, y1))
                            if len(slices) >= 6:
                                break
                        if len(slices) >= 6:
                            break

                    # Batch slice inference in groups of 4
                    batch_sz = 4
                    for b_i in range(0, len(slices), batch_sz):
                        sub_slices = slices[b_i:b_i + batch_sz]
                        tile_tensors = []
                        for (x0, y0, x1, y1) in sub_slices:
                            crop = rgb[y0:y1, x0:x1]
                            tile_tensors.append(TF.to_tensor(Image.fromarray(crop)).to(self.device))

                        with torch.no_grad():
                            tile_preds = self.model(tile_tensors)

                        for tp, (x0, y0, x1, y1) in zip(tile_preds, sub_slices):
                            tb = tp["boxes"].cpu().numpy()
                            ts = tp["scores"].cpu().numpy()
                            tl = tp["labels"].cpu().numpy()
                            for b, s, l in zip(tb, ts, tl):
                                if s >= 0.22:
                                    all_raw_boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
                                    all_raw_scores.append(float(s))
                                    all_raw_labels.append(int(l))

                # Precompute HSV for auto-rickshaw detection
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask_yellow = cv2.inRange(hsv, (14, 55, 55), (38, 255, 255))

                traffic_coco_map = {
                    "car": ("car", 0),
                    "motorcycle": ("motorcycle", 1),
                    "bus": ("bus", 3),
                    "truck": ("truck", 4),
                    "bicycle": ("bicycle", 5),
                    "person": ("pedestrian", 6)
                }

                # Step 1: Collect candidates separated into Vehicles vs Persons
                veh_boxes = []
                veh_scores = []
                veh_cats = []
                veh_cat_ids = []

                person_boxes = []
                person_scores = []

                cand_thresh = max(0.22, effective_thresh * 0.70)

                for b, s, l in zip(all_raw_boxes, all_raw_scores, all_raw_labels):
                    if s < cand_thresh:
                        continue
                    if l >= len(self.coco_categories):
                        continue
                    cat_name = self.coco_categories[l]
                    if cat_name not in traffic_coco_map:
                        continue

                    x1 = float(max(0.0, min(float(w - 1), b[0])))
                    y1 = float(max(0.0, min(float(h - 1), b[1])))
                    x2 = float(max(x1 + 1, min(float(w), b[2])))
                    y2 = float(max(y1 + 1, min(float(h), b[3])))

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 4 or bh <= 4:
                        continue

                    if cat_name == "person":
                        person_boxes.append([x1, y1, x2, y2])
                        person_scores.append(float(s))
                    else:
                        veh_boxes.append([x1, y1, x2, y2])
                        veh_scores.append(float(s))
                        veh_cats.append(cat_name)
                        veh_cat_ids.append(traffic_coco_map[cat_name][1])

                # Step 2: Class-Aware Vehicle NMS (Vehicles never suppressed by pedestrians)
                kept_vehicles = []
                if veh_boxes:
                    t_vb = torch.tensor(np.array(veh_boxes), dtype=torch.float32)
                    t_vs = torch.tensor(np.array(veh_scores), dtype=torch.float32)
                    t_vc = torch.tensor(np.array(veh_cat_ids), dtype=torch.int64)
                    keep_v = ops.batched_nms(t_vb, t_vs, t_vc, iou_threshold=0.45).numpy()

                    # Cross-vehicle deduplication (e.g. car vs truck duplicates)
                    sub_b = t_vb[keep_v]
                    sub_s = t_vs[keep_v]
                    keep_v_cross = ops.nms(sub_b, sub_s, iou_threshold=0.55).numpy()

                    for idx in keep_v_cross:
                        orig_idx = keep_v[idx]
                        kept_vehicles.append({
                            "box": veh_boxes[orig_idx],
                            "score": veh_scores[orig_idx],
                            "cat": veh_cats[orig_idx]
                        })

                # Step 3: Pedestrian NMS
                kept_persons = []
                if person_boxes:
                    t_pb = torch.tensor(np.array(person_boxes), dtype=torch.float32)
                    t_ps = torch.tensor(np.array(person_scores), dtype=torch.float32)
                    keep_p = ops.nms(t_pb, t_ps, iou_threshold=0.40).numpy()
                    for idx in keep_p:
                        kept_persons.append({
                            "box": person_boxes[idx],
                            "score": person_scores[idx],
                            "absorbed": False
                        })

                # Step 4: Rider-to-Two-Wheeler Fusion
                # When a person sits on a motorcycle or bicycle, merge them into the vehicle and absorb the pedestrian
                for v in kept_vehicles:
                    if v["cat"] in ["motorcycle", "bicycle"]:
                        vx1, vy1, vx2, vy2 = v["box"]
                        for p in kept_persons:
                            if p["absorbed"]:
                                continue
                            px1, py1, px2, py2 = p["box"]
                            # Horizontal overlap with bike
                            inter_x = max(0.0, min(vx2, px2) - max(vx1, px1))
                            pw = px2 - px1
                            # Vertical relation: person sits on or directly above bike
                            if inter_x > 0.20 * pw and (py1 <= vy2 and vy2 >= py1 + 0.25 * (py2 - py1)):
                                # Absorb rider into vehicle envelope
                                v["box"] = [
                                    min(vx1, px1),
                                    min(vy1, py1),
                                    max(vx2, px2),
                                    max(vy2, py2)
                                ]
                                v["score"] = max(v["score"], p["score"])
                                p["absorbed"] = True

                # Step 5: Carriageway Rider Disambiguation
                # In Indian traffic, any person detected in the driving lanes (not on sidewalk) is a rider on a 2-wheeler
                for p in kept_persons:
                    if p["absorbed"]:
                        continue
                    px1, py1, px2, py2 = p["box"]
                    pcx = (px1 + px2) / 2.0
                    pcy = (py1 + py2) / 2.0
                    pbw = px2 - px1
                    pbh = py2 - py1
                    paspect = pbw / max(1.0, pbh)

                    # In traffic carriageway: middle 85% of road width and below horizon
                    is_in_roadway = (pcy > h * 0.24) and (0.07 * w < pcx < 0.93 * w)
                    # Check if on sidewalk curb (sidewalks are on far edges with bollards / pedestrians)
                    is_on_curb = (pcx <= 0.08 * w) or (pcx >= 0.90 * w)

                    if is_in_roadway and not is_on_curb:
                        # Reclassify as motorcycle rider on the road
                        # Expand downward slightly to encompass the motorcycle body & tires
                        expanded_y2 = min(float(h), py2 + pbh * 0.28)
                        kept_vehicles.append({
                            "box": [px1, py1, px2, expanded_y2],
                            "score": p["score"],
                            "cat": "motorcycle"
                        })
                        p["absorbed"] = True

                # Step 6: Process and Classify Final Detections
                # (Perspective Sizing, Auto-Rickshaw, and Emergency Vehicle Ambulance Detection)
                for v in kept_vehicles:
                    box = v["box"]
                    s = v["score"]
                    c = v["cat"]

                    if s < effective_thresh:
                        continue

                    x1, y1, x2, y2 = box
                    bw = x2 - x1
                    bh = y2 - y1
                    area = bw * bh
                    aspect = bw / max(1.0, bh)
                    cy = (y1 + y2) / 2.0

                    expected_car_area = (400.0 + (cy / max(1.0, float(h))) * 700.0) * (w * h / (568.0 * 295.0))

                    # Crop vehicle for color and livery analysis
                    crop_y1, crop_y2 = max(0, int(y1)), min(h, int(y2))
                    crop_x1, crop_x2 = max(0, int(x1)), min(w, int(x2))
                    veh_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

                    target_name, target_id = traffic_coco_map[c]

                    # Auto-Rickshaw Livery Analysis (Yellow Body)
                    crop_yellow = mask_yellow[crop_y1:crop_y2, crop_x1:crop_x2]
                    yellow_ratio = float((crop_yellow > 0).mean()) if crop_yellow.size > 0 else 0.0

                    if c == "bus":
                        if area < (expected_car_area * 1.70) and s < 0.75:
                            if yellow_ratio > 0.14 and area <= expected_car_area:
                                target_name = "auto_rickshaw"
                                target_id = 2
                            else:
                                target_name = "car"
                                target_id = 0

                    if target_name in ["car", "truck", "motorcycle"]:
                        if yellow_ratio > 0.14 and 0.60 <= aspect <= 1.45:
                            if area <= (expected_car_area * 1.15):
                                target_name = "auto_rickshaw"
                                target_id = 2
                            else:
                                target_name = "truck"
                                target_id = 4

                    if target_name == "car" and (aspect < 0.65 or area < expected_car_area * 0.40) and cy < h * 0.55:
                        target_name = "motorcycle"
                        target_id = 1

                    # Emergency Vehicle / Ambulance Detection
                    # Inspect roof beacon and white/red livery on car, truck, bus, or van crops
                    if target_name in ["car", "truck", "bus"] and veh_crop.shape[0] >= 18 and veh_crop.shape[1] >= 18:
                        crop_hsv = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2HSV)
                        ch, cw = crop_hsv.shape[:2]

                        # Roofline region (upper 25% of vehicle)
                        roof_hsv = crop_hsv[:int(ch * 0.25), :]
                        m_r_red = cv2.inRange(roof_hsv, (0, 110, 85), (10, 255, 255)) | cv2.inRange(roof_hsv, (170, 110, 85), (180, 255, 255))
                        m_r_blue = cv2.inRange(roof_hsv, (95, 110, 85), (135, 255, 255))
                        m_r_amber = cv2.inRange(roof_hsv, (15, 120, 100), (32, 255, 255))
                        beacon_pix = cv2.countNonZero(m_r_red | m_r_blue | m_r_amber)
                        blue_pix = cv2.countNonZero(m_r_blue)

                        # Full vehicle livery: White body + Red cross/stripe
                        # Distinguish ambulance markings from normal passenger car red taillights:
                        # In an ambulance, emergency markings (red crosses, stripes, lettering) appear on the upper body and roof.
                        # In standard cars, red is strictly localized at the bottom rear taillights.
                        m_white = cv2.inRange(crop_hsv, (0, 0, 160), (180, 50, 255))
                        white_ratio = cv2.countNonZero(m_white) / float(ch * cw)

                        upper_hsv = crop_hsv[:int(ch * 0.55), :]
                        m_upper_red = cv2.inRange(upper_hsv, (0, 110, 80), (10, 255, 255)) | cv2.inRange(upper_hsv, (170, 110, 80), (180, 255, 255))
                        upper_red_ratio = cv2.countNonZero(m_upper_red) / max(1.0, float(upper_hsv.shape[0] * upper_hsv.shape[1]))

                        has_roof_beacon = (blue_pix >= 6) or (beacon_pix >= 14 and (beacon_pix / max(1, roof_hsv.shape[0] * roof_hsv.shape[1])) >= 0.015)
                        has_ambulance_livery = (white_ratio >= 0.22 and upper_red_ratio >= 0.016 and aspect <= 1.35)

                        if has_roof_beacon or has_ambulance_livery:
                            target_name = "emergency_vehicle"
                            target_id = 7
                            s = max(float(s), 0.95)

                    detections.append(BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(s),
                        class_id=target_id,
                        class_name=target_name
                    ))

                # Step 7: Append genuine pedestrians (on sidewalk or zebra crossing)
                for p in kept_persons:
                    if p["absorbed"] or p["score"] < effective_thresh:
                        continue
                    px1, py1, px2, py2 = p["box"]
                    detections.append(BoundingBox(
                        x1=px1, y1=py1, x2=px2, y2=py2,
                        confidence=float(p["score"]),
                        class_id=6,
                        class_name="pedestrian"
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
