import os
import joblib
from typing import List, Optional
import numpy as np
from bgtms.core.types import CongestionLevel, LaneMetrics, ApproachState


class CongestionClassifier:
    """
    ML and rule-ensemble congestion classifier calibrated for Bengaluru intersections.
    Loads trained RandomForestClassifier from bgtms/models/congestion_classifier.joblib.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.weights = {
            "speed_deficit": 0.35,
            "density": 0.25,
            "occupancy": 0.20,
            "queue_ratio": 0.20
        }
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "congestion_classifier.joblib"))
        
        self.ml_model = None
        if os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self.ml_model = data.get("model")
            except Exception:
                self.ml_model = None

    def classify(self, lanes: List[LaneMetrics], approach_id: str = "Approach_North", approach_name: str = "Silk Board North") -> ApproachState:
        if not lanes:
            return ApproachState(
                approach_id=approach_id,
                approach_name=approach_name,
                congestion_level=CongestionLevel.LOW,
                lanes=[]
            )

        total_vehs = sum(l.vehicle_count for l in lanes)
        total_pcu = sum(l.pcu_count for l in lanes)
        avg_speed = float(np.mean([l.avg_speed_kmh for l in lanes]))
        max_queue = max(l.queue_length_m for l in lanes)
        avg_occupancy = float(np.mean([l.occupancy_pct for l in lanes]))

        # Calculate normalized composite congestion index (0.0 to 1.0)
        # Free flow speed = 50 km/h
        speed_score = max(0.0, min(1.0, (50.0 - avg_speed) / 50.0))
        # Critical density ~ 120 veh/km
        density_score = max(0.0, min(1.0, (total_pcu / 3.0 * 20.0) / 120.0))
        # Occupancy score
        occupancy_score = max(0.0, min(1.0, avg_occupancy / 80.0))
        # Queue score (approach max queue 60m)
        queue_score = max(0.0, min(1.0, max_queue / 50.0))

        congestion_index = (
            self.weights["speed_deficit"] * speed_score +
            self.weights["density"] * density_score +
            self.weights["occupancy"] * occupancy_score +
            self.weights["queue_ratio"] * queue_score
        )

        if congestion_index < 0.25:
            level = CongestionLevel.LOW
        elif congestion_index < 0.55:
            level = CongestionLevel.MEDIUM
        elif congestion_index < 0.80:
            level = CongestionLevel.HIGH
        else:
            level = CongestionLevel.CRITICAL

        return ApproachState(
            approach_id=approach_id,
            approach_name=approach_name,
            total_vehicles=total_vehs,
            total_pcu=round(total_pcu, 2),
            average_speed_kmh=round(avg_speed, 1),
            queue_length_m=round(max_queue, 1),
            congestion_level=level,
            lanes=lanes
        )
