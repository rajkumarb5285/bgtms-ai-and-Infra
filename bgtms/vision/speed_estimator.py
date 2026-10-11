"""
BGTMS - Speed Estimation Module (M1-13)
Estimates real-world vehicle speed in km/h from trajectories and calibrated perspective homography.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from bgtms.core.types import TrackedVehicle


class SpeedEstimator:
    """
    Computes vehicle ground-plane speed using 4-point camera perspective homography.
    """

    def __init__(self, src_points: Optional[List[List[float]]] = None,
                 dst_points: Optional[List[List[float]]] = None,
                 fps: float = 30.0):
        self.fps = fps
        # Default Silk Board CCTV calibration:
        # Source pixels on 1920x1080 -> Destination meters (15m width x 45m length road sector)
        self.src = np.float32(src_points or [[450, 600], [1470, 600], [1820, 1020], [100, 1020]])
        self.dst = np.float32(dst_points or [[0, 0], [15, 0], [15, 45], [0, 45]])

        self.H = cv2.getPerspectiveTransform(self.src, self.dst)
        self.speed_cache: Dict[int, float] = {}

    def pixel_to_meters(self, pixel_pt: Tuple[float, float]) -> Tuple[float, float]:
        px = np.array([[[pixel_pt[0], pixel_pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(px, self.H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def estimate_speed(self, vehicle: TrackedVehicle, window_size: int = 6) -> float:
        """
        Calculates metric speed in km/h using recent trajectory points.
        """
        traj = vehicle.trajectory
        if len(traj) < 3:
            return self.speed_cache.get(vehicle.track_id, 0.0)

        # Use recent window of points for instantaneous velocity
        recent = traj[-min(len(traj), window_size):]
        p_start = self.pixel_to_meters(recent[0])
        p_end = self.pixel_to_meters(recent[-1])

        # Euclidean distance in meters
        dist_m = np.sqrt((p_end[0] - p_start[0])**2 + (p_end[1] - p_start[1])**2)
        dt_sec = (len(recent) - 1) / self.fps

        if dt_sec <= 0:
            return 0.0

        raw_speed_mps = dist_m / dt_sec
        raw_speed_kmh = raw_speed_mps * 3.6

        # EMA smoothing
        prev = self.speed_cache.get(vehicle.track_id, raw_speed_kmh)
        smoothed = 0.7 * prev + 0.3 * raw_speed_kmh
        
        # Physical clamp for urban roads (0 to 120 km/h)
        smoothed = max(0.0, min(120.0, smoothed))
        self.speed_cache[vehicle.track_id] = smoothed
        vehicle.speed_kmh = round(smoothed, 1)

        return vehicle.speed_kmh
