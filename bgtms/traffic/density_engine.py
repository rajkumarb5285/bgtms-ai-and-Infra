"""
BGTMS - Traffic Density Engine (M1-15)
Computes per-lane vehicle count, density (veh/km), occupancy %,
flow rates (veh/hr & PCU/hr) with Indian Roads Congress PCU equivalencies.
"""

from typing import List, Dict, Optional
import numpy as np
from bgtms.core.types import TrackedVehicle, LaneMetrics, ApproachState
from bgtms.core.config import get_classes


class TrafficDensityEngine:
    """
    Real-time microscopic traffic density and PCU flow calculator.
    """

    def __init__(self, approach_id: str = "Approach_North", approach_length_m: float = 45.0):
        self.approach_id = approach_id
        self.approach_length_m = approach_length_m
        
        cfg = get_classes()
        self.pcu_map = {int(k): v["pcu"] for k, v in cfg.get("classes", {}).items()}

    def compute_lane_metrics(self, vehicles: List[TrackedVehicle], num_lanes: int = 3) -> List[LaneMetrics]:
        metrics: List[LaneMetrics] = []

        for lane_id in range(1, num_lanes + 1):
            lane_vehs = [v for v in vehicles if v.lane_id == lane_id]
            count = len(lane_vehs)
            
            # Sum PCU
            pcu_sum = sum(v.pcu for v in lane_vehs)

            # Density: vehicles per kilometer = (count / length_in_meters) * 1000
            density_km = (pcu_sum / max(1.0, self.approach_length_m)) * 1000.0

            # Speeds
            speeds = [v.speed_kmh for v in lane_vehs]
            avg_speed = float(np.mean(speeds)) if speeds else 40.0

            # Occupancy %: estimated physical road space occupied
            # Average vehicle footprint: car=4m, bike=1.8m, auto=2.6m, bus=11m
            footprint_sum = sum(self._get_footprint(v.class_id) for v in lane_vehs)
            occupancy_pct = min(100.0, (footprint_sum / max(1.0, self.approach_length_m)) * 100.0)

            # Flow: q = k * v (fundamental traffic flow relation)
            flow_pcu_hr = density_km * avg_speed

            # Waiting vehicles
            waiting = sum(1 for v in lane_vehs if v.speed_kmh < 3.0)

            metrics.append(LaneMetrics(
                lane_id=lane_id,
                approach_id=self.approach_id,
                vehicle_count=count,
                pcu_count=round(pcu_sum, 2),
                density_veh_km=round(density_km, 1),
                occupancy_pct=round(occupancy_pct, 1),
                flow_pcu_hr=round(flow_pcu_hr, 1),
                avg_speed_kmh=round(avg_speed, 1),
                queue_length_m=0.0, # Filled by queue detector
                waiting_vehicles=waiting
            ))

        return metrics

    def _get_footprint(self, class_id: int) -> float:
        # Physical length in meters
        footprints = {
            0: 4.2,  # car
            1: 1.8,  # motorcycle
            2: 2.6,  # auto rickshaw
            3: 11.5, # bus
            4: 8.5,  # truck
            5: 1.6,  # bicycle
            6: 0.5,  # pedestrian
            7: 5.5   # emergency
        }
        return footprints.get(class_id, 4.0)
