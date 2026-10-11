"""
BGTMS - Traffic Violation Detection Engine (M1-20)
Detects red-light violations, wrong-way driving, and stop-line encroachments.
"""

import time
import uuid
from typing import List, Optional
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType


class ViolationDetector:
    """
    Automated traffic law enforcement and violation monitoring.
    """

    def __init__(self, stop_line_y: float = 880.0):
        self.stop_line_y = stop_line_y
        self.reported_violations: set = set()

    def check_violations(self, vehicles: List[TrackedVehicle], is_red_light: bool, current_time: Optional[float] = None) -> List[IncidentAlert]:
        now = current_time or time.time()
        violations: List[IncidentAlert] = []

        for v in vehicles:
            v_key = f"{v.track_id}_{v.class_name}"

            # 1. Red-Light Running: Crossing stop-line with forward velocity during red phase
            if is_red_light and v_key not in self.reported_violations:
                if v.bbox.y2 > self.stop_line_y and v.speed_kmh > 5.0:
                    # Check if previous point was behind stop line
                    if len(v.trajectory) >= 2 and v.trajectory[-2][1] <= self.stop_line_y:
                        alert = IncidentAlert(
                            incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                            incident_type=IncidentType.RED_LIGHT_VIOLATION,
                            severity="CRITICAL",
                            timestamp=now,
                            track_id=v.track_id,
                            approach_id=v.approach_id,
                            location=v.bbox.center,
                            description=f"Red-Light Jump: {v.class_name} #{v.track_id} crossed stop-line at {v.speed_kmh} km/h",
                            details={"speed_kmh": v.speed_kmh, "stop_line_y": self.stop_line_y}
                        )
                        violations.append(alert)
                        self.reported_violations.add(v_key)

            # 2. Wrong-Way Driving: Trajectory vector inverted (moving upwards on approach where normal flow is downwards)
            if len(v.trajectory) >= 8 and f"ww_{v_key}" not in self.reported_violations:
                dy = v.trajectory[-1][1] - v.trajectory[0][1]
                if dy < -40.0 and v.speed_kmh > 8.0:
                    alert = IncidentAlert(
                        incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.WRONG_WAY,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=v.approach_id,
                        location=v.bbox.center,
                        description=f"Dangerous Wrong-Way vehicle detected: #{v.track_id} ({v.class_name})",
                        details={"heading_delta_y": dy, "speed_kmh": v.speed_kmh}
                    )
                    violations.append(alert)
                    self.reported_violations.add(f"ww_{v_key}")

        return violations
