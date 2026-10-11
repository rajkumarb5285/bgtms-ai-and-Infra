"""
BGTMS - Lane Detection & Vehicle Association (M1-14)
Defines approach lane ROIs and associates tracked vehicles with specific lanes.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from bgtms.core.types import TrackedVehicle


class LaneDetector:
    """
    Manages polygonal lane ROIs and maps tracked vehicle centers to lanes.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080):
        self.w = frame_width
        self.h = frame_height

        # Define 3 lanes for approaching arm (e.g. Left Turn/Straight, Central Straight, Right Turn)
        self.lane_polygons: Dict[int, np.ndarray] = {
            1: np.array([[100, 1020], [450, 600], [790, 600], [670, 1020]], dtype=np.int32),   # Lane 1 (Left)
            2: np.array([[670, 1020], [790, 600], [1130, 600], [1240, 1020]], dtype=np.int32), # Lane 2 (Center)
            3: np.array([[1240, 1020], [1130, 600], [1470, 600], [1820, 1020]], dtype=np.int32) # Lane 3 (Right)
        }

    def associate_vehicle_to_lane(self, vehicle: TrackedVehicle, frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> Optional[int]:
        w = frame_w if frame_w else self.w
        h = frame_h if frame_h else self.h

        # Test vehicle bottom-center point (contact patch on road)
        bx = (vehicle.bbox.x1 + vehicle.bbox.x2) / 2.0
        by = vehicle.bbox.y2
        pt = (float(bx), float(by))

        # Check polygons with resolution scaling
        sx = w / float(self.w)
        sy = h / float(self.h)
        for lane_id, poly in self.lane_polygons.items():
            scaled_poly = (poly.astype(float) * [sx, sy]).astype(np.int32)
            dist = cv2.pointPolygonTest(scaled_poly, pt, False)
            if dist >= 0:
                vehicle.lane_id = lane_id
                return lane_id

        # Fallback to realistic horizontal road partitions:
        # Lane 1: Left 35% of carriageway
        # Lane 2: Center 33% of carriageway
        # Lane 3: Right 32% of carriageway
        norm_x = bx / max(1.0, float(w))
        if norm_x < 0.35:
            lane_id = 1
        elif norm_x < 0.68:
            lane_id = 2
        else:
            lane_id = 3

        vehicle.lane_id = lane_id
        return vehicle.lane_id


    def draw_lanes(self, image: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        overlay = image.copy()
        colors = {
            1: (255, 150, 50),
            2: (50, 200, 50),
            3: (50, 100, 255)
        }
        for lane_id, poly in self.lane_polygons.items():
            color = colors.get(lane_id, (200, 200, 200))
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(overlay, [poly], True, color, 2)
            
            # Label
            moment = cv2.moments(poly)
            if moment["m00"] != 0:
                cx = int(moment["m10"] / moment["m00"])
                cy = int(moment["m01"] / moment["m00"])
                cv2.putText(overlay, f"LANE {lane_id}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
