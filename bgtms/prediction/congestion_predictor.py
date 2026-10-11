"""
BGTMS - Predictive AI Congestion Forecaster (M1-18)
Predicts traffic congestion levels, expected queue lengths,
and PCU flow across 5-min, 15-min, and 30-min horizons.
"""

from typing import Dict, List, Any
import numpy as np
from bgtms.core.types import ApproachState, CongestionLevel


class CongestionPredictor:
    """
    Time-series forecasting model for short-to-medium horizon traffic state prediction.
    """

    def __init__(self, history_len: int = 60):
        self.history_len = history_len
        self.pcu_history: List[float] = []
        self.queue_history: List[float] = []
        self.speed_history: List[float] = []

    def record_step(self, approach_state: ApproachState):
        self.pcu_history.append(approach_state.total_pcu)
        self.queue_history.append(approach_state.queue_length_m)
        self.speed_history.append(approach_state.average_speed_kmh)

        if len(self.pcu_history) > self.history_len:
            self.pcu_history.pop(0)
            self.queue_history.pop(0)
            self.speed_history.pop(0)

    def predict(self) -> Dict[str, Any]:
        """
        Generates predictions for 5, 15, and 30 minute horizons.
        """
        curr_pcu = self.pcu_history[-1] if self.pcu_history else 20.0
        curr_q = self.queue_history[-1] if self.queue_history else 15.0
        curr_spd = self.speed_history[-1] if self.speed_history else 35.0

        # Calculate arrival rate derivative (trend)
        trend_pcu = 0.0
        if len(self.pcu_history) >= 5:
            trend_pcu = float(np.polyfit(range(5), self.pcu_history[-5:], 1)[0])

        horizons = [5, 15, 30]
        forecasts = {}

        for h in horizons:
            # Compounding dampener for long horizon stability
            damping = np.exp(-0.03 * h)
            pred_pcu = max(2.0, curr_pcu + (trend_pcu * h * 0.4 * damping))
            pred_queue = max(0.0, curr_q + (trend_pcu * 1.5 * h * damping))
            pred_speed = max(5.0, min(60.0, curr_spd - (trend_pcu * 0.8 * h * damping)))

            # Forecast congestion level
            if pred_pcu > 45.0 or pred_queue > 40.0:
                level = CongestionLevel.CRITICAL
            elif pred_pcu > 30.0 or pred_queue > 25.0:
                level = CongestionLevel.HIGH
            elif pred_pcu > 18.0:
                level = CongestionLevel.MEDIUM
            else:
                level = CongestionLevel.LOW

            forecasts[f"{h}_min"] = {
                "horizon_minutes": h,
                "predicted_pcu": round(pred_pcu, 1),
                "predicted_queue_m": round(pred_queue, 1),
                "predicted_speed_kmh": round(pred_speed, 1),
                "congestion_level": level.value,
                "bottleneck_risk_pct": round(min(100.0, (pred_queue / 50.0) * 100.0), 1)
            }

        return forecasts
