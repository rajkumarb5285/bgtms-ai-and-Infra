"""
BGTMS - Multi-Object Tracking Pipeline (M1-12)
ByteTrack / Kalman-Filter based persistent vehicle tracking,
trajectory history, and CSV export.
"""

import os
import csv
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from bgtms.core.types import BoundingBox, TrackedVehicle
from bgtms.core.config import get_classes


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: BoundingBox):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_id = bbox.class_id
        self.class_name = bbox.class_name
        
        # State: [x_center, y_center, s (area), r (ratio), vx, vy, vs]
        xc, yc = bbox.center
        w = max(1.0, bbox.width)
        h = max(1.0, bbox.height)
        s = w * h
        r = w / float(h)
        self.state = np.array([xc, yc, s, r, 0, 0, 0], dtype=float)

        self.time_since_update = 0
        self.history: List[Tuple[float, float]] = [(xc, yc)]
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox
        self.speed_kmh = 0.0

    def update(self, bbox: BoundingBox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.age += 1

        xc, yc = bbox.center
        w = max(1.0, bbox.width)
        h = max(1.0, bbox.height)
        s = w * h
        r = w / float(h)

        # Simple alpha-beta velocity estimation
        vx = xc - self.state[0]
        vy = yc - self.state[1]

        self.state = np.array([xc, yc, s, r, vx, vy, 0], dtype=float)
        self.history.append((xc, yc))
        if len(self.history) > 300:
            self.history.pop(0)
            
        self.last_bbox = bbox

    def predict(self) -> np.ndarray:
        # Constant velocity model
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_bbox()

    def get_bbox(self) -> np.ndarray:
        xc, yc, s, r = self.state[0], self.state[1], self.state[2], self.state[3]
        w = np.sqrt(max(1.0, s * r))
        h = max(1.0, s / max(1e-5, w))
        return np.array([xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0])


class MultiObjectTracker:
    """
    ByteTrack tracking manager for Bengaluru traffic streams.
    Implements two-stage bipartite association:
    Stage 1: High-confidence detections with active tracks.
    Stage 2: Low-confidence detections with remaining tracks for occlusion recovery.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
        high_score_thresh: float = 0.30,
        low_score_thresh: float = 0.15,
        iou_low_threshold: float = 0.20
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.iou_low_threshold = iou_low_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        cfg = get_classes()
        self.pcu_map = {int(k): v["pcu"] for k, v in cfg.get("classes", {}).items()}

    def update(self, detections: List[BoundingBox]) -> List[TrackedVehicle]:
        self.frame_count += 1

        # 1. Predict new locations of existing trackers
        for trk in self.trackers:
            trk.predict()

        # Separate detections into high and low confidence (ByteTrack)
        dets_high: List[BoundingBox] = []
        dets_low: List[BoundingBox] = []
        for d in detections:
            if d.confidence >= self.high_score_thresh:
                dets_high.append(d)
            elif d.confidence >= self.low_score_thresh:
                dets_low.append(d)

        # Fallback if all detections have uniform or low confidence
        if not dets_high and detections:
            dets_high = list(detections)

        # Stage 1: Associate high confidence detections with all trackers
        matched_1, unmatched_dets_1, unmatched_trks_1 = self._associate(
            dets_high, self.trackers, self.iou_threshold
        )
        for d_idx, t_idx in matched_1:
            self.trackers[t_idx].update(dets_high[d_idx])

        # Stage 2: Associate remaining unmatched trackers with low confidence detections (occlusion recovery)
        if unmatched_trks_1 and dets_low:
            sub_trackers = [self.trackers[i] for i in unmatched_trks_1]
            matched_2, _, _ = self._associate(
                dets_low, sub_trackers, self.iou_low_threshold
            )
            for d_idx, sub_t_idx in matched_2:
                real_t_idx = unmatched_trks_1[sub_t_idx]
                self.trackers[real_t_idx].update(dets_low[d_idx])

        # Stage 3: Create new trackers for unmatched high confidence detections
        for d_idx in unmatched_dets_1:
            trk = KalmanBoxTracker(dets_high[d_idx])
            self.trackers.append(trk)

        # Filter active tracks and prune dead ones
        active_vehicles: List[TrackedVehicle] = []
        remaining_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Valid confirmed track
                is_emergency = trk.class_id == 7
                pcu_val = self.pcu_map.get(trk.class_id, 1.0)
                
                active_vehicles.append(TrackedVehicle(
                    track_id=trk.id,
                    class_id=trk.class_id,
                    class_name=trk.class_name,
                    pcu=pcu_val,
                    bbox=trk.last_bbox,
                    trajectory=list(trk.history),
                    speed_kmh=trk.speed_kmh,
                    is_emergency=is_emergency
                ))

            if trk.time_since_update < self.max_age:
                remaining_trackers.append(trk)

        self.trackers = remaining_trackers
        return active_vehicles

    def _associate(self, detections: List[BoundingBox], trackers: List[KalmanBoxTracker], iou_thresh: Optional[float] = None):
        if iou_thresh is None:
            iou_thresh = self.iou_threshold
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=float)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk.get_bbox())

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        # Greedy bipartite matching
        while True:
            if len(unmatched_dets) == 0 or len(unmatched_trks) == 0:
                break
            max_iou = 0.0
            best_d = -1
            best_t = -1
            for d in unmatched_dets:
                for t in unmatched_trks:
                    if iou_matrix[d, t] > max_iou:
                        max_iou = iou_matrix[d, t]
                        best_d = d
                        best_t = t

            if max_iou >= iou_thresh:
                matched.append((best_d, best_t))
                unmatched_dets.remove(best_d)
                unmatched_trks.remove(best_t)
            else:
                break

        return matched, unmatched_dets, unmatched_trks

    def _iou(self, det: BoundingBox, trk_box: np.ndarray) -> float:
        x1 = max(det.x1, trk_box[0])
        y1 = max(det.y1, trk_box[1])
        x2 = min(det.x2, trk_box[2])
        y2 = min(det.y2, trk_box[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_det = det.area
        area_trk = max(1.0, (trk_box[2] - trk_box[0]) * (trk_box[3] - trk_box[1]))
        union = area_det + area_trk - inter
        return inter / union if union > 0 else 0.0

    def export_trajectories_to_csv(self, filepath: str):
        """Export all recorded trajectories to CSV format (M1-12 Deliverable)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "class_id", "class_name", "frame_idx", "x_pixel", "y_pixel", "speed_kmh"])
            for trk in self.trackers:
                for idx, pt in enumerate(trk.history):
                    writer.writerow([trk.id, trk.class_id, trk.class_name, idx, pt[0], pt[1], trk.speed_kmh])
        print(f"[+] Trajectories exported to: {filepath}")
