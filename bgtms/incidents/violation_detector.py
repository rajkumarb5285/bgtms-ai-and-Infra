"""
BGTMS - Traffic Violation Detection Engine (M1-20 / M2)
Detects red-light violations, stop-line encroachments, wrong-way driving, and speeding.
"""

import time
import uuid
from typing import List, Optional, Set, Dict
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType


class ViolationDetector:
    """
    Automated traffic law enforcement and safety violation monitoring.
    """

    def __init__(
        self,
        stop_line_y: float = 880.0,
        encroach_buffer_px: float = 40.0,
        speed_limit_kmh: float = 50.0,
    ):
        self.stop_line_y = stop_line_y
        self.encroach_buffer_px = encroach_buffer_px
        self.speed_limit_kmh = speed_limit_kmh
        self.reported_violations: Set[str] = set()

    def check_violations(
        self,
        vehicles: List[TrackedVehicle],
        is_red_light: bool,
        current_time: Optional[float] = None,
    ) -> List[IncidentAlert]:
        """
        Evaluate traffic violations across active tracked vehicles.
        """
        now = current_time or time.time()
        violations: List[IncidentAlert] = []

        for v in vehicles:
            v_key = f"{v.track_id}_{v.class_name}"

            # 1. Red-Light Jumping: Crossing stop-line with forward velocity during red phase
            if is_red_light:
                red_jump_key = f"rl_{v_key}"
                if red_jump_key not in self.reported_violations:
                    if v.bbox.y2 > (self.stop_line_y + self.encroach_buffer_px) and v.speed_kmh > 5.0:
                        # Check previous trajectory point was behind stop line
                        if len(v.trajectory) >= 2 and v.trajectory[-2][1] <= self.stop_line_y:
                            alert = IncidentAlert(
                                incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                                incident_type=IncidentType.RED_LIGHT_VIOLATION,
                                severity="CRITICAL",
                                timestamp=now,
                                track_id=v.track_id,
                                approach_id=v.approach_id,
                                location=v.bbox.center,
                                description=f"Red-Light Jump: {v.class_name} #{v.track_id} crossed stop-line at {v.speed_kmh:.1f} km/h",
                                details={
                                    "speed_kmh": round(v.speed_kmh, 1),
                                    "stop_line_y": self.stop_line_y,
                                    "class_name": v.class_name,
                                },
                            )
                            violations.append(alert)
                            self.reported_violations.add(red_jump_key)

                # 2. Stop-Line / Zebra Crossing Encroachment
                encroach_key = f"enc_{v_key}"
                if encroach_key not in self.reported_violations and red_jump_key not in self.reported_violations:
                    if self.stop_line_y < v.bbox.y2 <= (self.stop_line_y + self.encroach_buffer_px):
                        if v.speed_kmh < 5.0:  # stopped on zebra crossing
                            alert = IncidentAlert(
                                incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                                incident_type=IncidentType.STOP_LINE_VIOLATION,
                                severity="WARNING",
                                timestamp=now,
                                track_id=v.track_id,
                                approach_id=v.approach_id,
                                location=v.bbox.center,
                                description=f"Stop-line encroachment: {v.class_name} #{v.track_id} occupying pedestrian crossing zone",
                                details={
                                    "speed_kmh": round(v.speed_kmh, 1),
                                    "stop_line_y": self.stop_line_y,
                                    "y2": round(v.bbox.y2, 1),
                                },
                            )
                            violations.append(alert)
                            self.reported_violations.add(encroach_key)

            # 3. Wrong-Way Driving (Direction vector opposed to standard approach lane flow)
            ww_key = f"ww_{v_key}"
            if len(v.trajectory) >= 6 and ww_key not in self.reported_violations:
                dy = v.trajectory[-1][1] - v.trajectory[0][1]
                if dy < -35.0 and v.speed_kmh > 7.0:
                    alert = IncidentAlert(
                        incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.WRONG_WAY,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=v.approach_id,
                        location=v.bbox.center,
                        description=f"DANGEROUS WRONG-WAY DRIVING: {v.class_name} #{v.track_id} travelling against flow at {v.speed_kmh:.1f} km/h",
                        details={"heading_delta_y": round(dy, 1), "speed_kmh": round(v.speed_kmh, 1)},
                    )
                    violations.append(alert)
                    self.reported_violations.add(ww_key)

            # 4. Urban Speeding Violation (> 50 km/h in junction approach zone)
            spd_key = f"spd_{v_key}"
            if v.speed_kmh > self.speed_limit_kmh and spd_key not in self.reported_violations:
                alert = IncidentAlert(
                    incident_id=f"VIO-{uuid.uuid4().hex[:6].upper()}",
                    incident_type=IncidentType.RED_LIGHT_VIOLATION,  # Or general violation
                    severity="WARNING",
                    timestamp=now,
                    track_id=v.track_id,
                    approach_id=v.approach_id,
                    location=v.bbox.center,
                    description=f"Over-speeding: {v.class_name} #{v.track_id} travelling at {v.speed_kmh:.1f} km/h (Limit: {self.speed_limit_kmh:.0f} km/h)",
                    details={"speed_kmh": round(v.speed_kmh, 1), "speed_limit_kmh": self.speed_limit_kmh},
                )
                violations.append(alert)
                self.reported_violations.add(spd_key)

        return violations

    def reset(self) -> None:
        """Reset reported violations cache."""
        self.reported_violations.clear()
