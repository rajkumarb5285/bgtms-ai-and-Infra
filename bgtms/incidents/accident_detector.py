"""
BGTMS - Accident & Incident Detection Engine (M1-19)
Detects collisions, sudden decelerations, stalled vehicles, and road blockages.
"""

import time
import uuid
from typing import List, Dict, Optional
import numpy as np
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType


class AccidentDetector:
    """
    Real-time incident detection using kinematic anomalies and spatial interactions.
    """

    def __init__(self, sudden_decel_thresh: float = 25.0, stalled_time_sec: float = 30.0, fps: float = 30.0):
        self.decel_thresh = sudden_decel_thresh
        self.stalled_frames_thresh = int(stalled_time_sec * fps)
        self.fps = fps
        self.recent_speeds: Dict[int, List[float]] = {}
        self.active_alerts: List[IncidentAlert] = []

    def check_incidents(self, vehicles: List[TrackedVehicle], current_time: Optional[float] = None) -> List[IncidentAlert]:
        now = current_time or time.time()
        new_alerts: List[IncidentAlert] = []

        # 1. Sudden Deceleration & Stalled Vehicle Check
        for v in vehicles:
            if v.track_id not in self.recent_speeds:
                self.recent_speeds[v.track_id] = []
            
            speeds = self.recent_speeds[v.track_id]
            speeds.append(v.speed_kmh)
            if len(speeds) > 15:
                speeds.pop(0)

            # Check sudden drop
            if len(speeds) >= 10:
                delta_v = speeds[0] - speeds[-1]
                if delta_v > self.decel_thresh and speeds[-1] < 5.0:
                    alert = IncidentAlert(
                        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.SUDDEN_DECELERATION,
                        severity="WARNING",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=v.approach_id,
                        location=v.bbox.center,
                        description=f"Sudden sharp deceleration detected ({delta_v:.1f} km/h drop)",
                        details={"initial_speed": speeds[0], "final_speed": speeds[-1]}
                    )
                    new_alerts.append(alert)

            # Stalled vehicle check (stopped on active carriageway)
            if v.speed_kmh < 1.0 and not v.in_queue:
                v.stopped_frames += 1
                if v.stopped_frames == self.stalled_frames_thresh:
                    alert = IncidentAlert(
                        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.STALLED_VEHICLE,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=v.approach_id,
                        location=v.bbox.center,
                        description=f"Stalled vehicle blocking lane {v.lane_id or 'road'}",
                        details={"stopped_duration_sec": self.stalled_frames_thresh / self.fps}
                    )
                    new_alerts.append(alert)
            else:
                v.stopped_frames = 0

        # 2. Collision Check: Overlapping bounding boxes between moving vehicles
        n = len(vehicles)
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = vehicles[i], vehicles[j]
                if self._check_collision_overlap(v1, v2):
                    alert = IncidentAlert(
                        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.COLLISION,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v1.track_id,
                        approach_id=v1.approach_id,
                        location=v1.bbox.center,
                        description=f"Potential collision between Vehicle #{v1.track_id} ({v1.class_name}) and #{v2.track_id} ({v2.class_name})",
                        details={"vehicle_1": v1.track_id, "vehicle_2": v2.track_id}
                    )
                    new_alerts.append(alert)

        self.active_alerts.extend(new_alerts)
        if len(self.active_alerts) > 50:
            self.active_alerts = self.active_alerts[-50:]

        return new_alerts

    def _check_collision_overlap(self, v1: TrackedVehicle, v2: TrackedVehicle) -> bool:
        # Bounding box IoU / intersection check
        x1 = max(v1.bbox.x1, v2.bbox.x1)
        y1 = max(v1.bbox.y1, v2.bbox.y1)
        x2 = min(v1.bbox.x2, v2.bbox.x2)
        y2 = min(v1.bbox.y2, v2.bbox.y2)

        if x2 > x1 and y2 > y1:
            inter_area = (x2 - x1) * (y2 - y1)
            min_area = min(v1.bbox.area, v2.bbox.area)
            if (inter_area / max(1.0, min_area)) > 0.45:
                # High overlap with abrupt stop indicates crash contact
                if v1.speed_kmh < 4.0 or v2.speed_kmh < 4.0:
                    return True
        return False
