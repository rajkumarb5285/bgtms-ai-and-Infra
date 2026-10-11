"""
BGTMS - Queue-Length Detection & Shockwave Analysis (M1-16)
Measures queue length, waiting vehicles, and shockwave propagation backwards from the stop-line.
"""

from typing import List, Dict, Tuple
import numpy as np
from bgtms.core.types import TrackedVehicle, LaneMetrics
from bgtms.vision.speed_estimator import SpeedEstimator


class QueueDetector:
    """
    Detects physical queue back-up and queue shockwave dynamics.
    """

    def __init__(self, stop_line_y: float = 880.0, speed_estimator: SpeedEstimator = None):
        self.stop_line_y = stop_line_y
        self.speed_estimator = speed_estimator or SpeedEstimator()
        self.prev_queue_lengths: Dict[int, float] = {}

    def compute_queue(self, lane_id: int, vehicles: List[TrackedVehicle], dt_sec: float = 1.0) -> Tuple[float, int, float]:
        """
        Returns: (queue_length_meters, waiting_vehicles_count, growth_rate_mps)
        """
        lane_vehs = [v for v in vehicles if v.lane_id == lane_id]
        
        # Vehicles stopped or crawling (speed < 3.5 km/h) approaching stop line
        stopped = [v for v in lane_vehs if v.speed_kmh < 3.5 and v.bbox.y2 <= self.stop_line_y + 100]
        
        if not stopped:
            self.prev_queue_lengths[lane_id] = 0.0
            return 0.0, 0, 0.0

        # Sort from stop-line backwards (decreasing y pixel)
        stopped.sort(key=lambda v: v.bbox.y2, reverse=True)

        # Stop line coordinate in meters
        stop_m = self.speed_estimator.pixel_to_meters((960, self.stop_line_y))
        
        # Furthest upstream stopped vehicle
        tail_veh = stopped[-1]
        tail_m = self.speed_estimator.pixel_to_meters(tail_veh.bbox.center)

        # Distance along approach (Y component in meters)
        queue_m = abs(tail_m[1] - stop_m[1])
        queue_m = max(5.0 * len(stopped), queue_m) # Physical sanity check (>= 5m per stopped vehicle)

        # Mark in_queue on vehicles
        for v in stopped:
            v.in_queue = True

        prev_q = self.prev_queue_lengths.get(lane_id, queue_m)
        growth_rate_mps = (queue_m - prev_q) / max(0.1, dt_sec)
        self.prev_queue_lengths[lane_id] = queue_m

        return round(queue_m, 1), len(stopped), round(growth_rate_mps, 2)
