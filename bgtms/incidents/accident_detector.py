"""
BGTMS - Accident & Incident Detection Engine (M1-19 / M2)
Detects collisions, sudden decelerations, stalled vehicles, and road blockages.
"""

import time
import uuid
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType


class AccidentDetector:
    """
    Real-time incident detection using kinematic anomalies, spatial interactions,
    and temporal tracking history.
    """

    def __init__(
        self,
        sudden_decel_thresh: float = 25.0,
        stalled_time_sec: float = 30.0,
        fps: float = 30.0,
        collision_iou_thresh: float = 0.40,
        cooldown_sec: float = 10.0,
    ):
        self.decel_thresh = sudden_decel_thresh
        self.stalled_frames_thresh = int(stalled_time_sec * fps)
        self.fps = fps
        self.collision_iou_thresh = collision_iou_thresh
        self.cooldown_sec = cooldown_sec
        self.recent_speeds: Dict[int, List[float]] = {}
        self.active_alerts: List[IncidentAlert] = []
        self.reported_collision_pairs: Dict[Tuple[int, int], float] = {}
        self.reported_stalls: Dict[int, float] = {}

    def check_incidents(
        self, vehicles: List[TrackedVehicle], current_time: Optional[float] = None
    ) -> List[IncidentAlert]:
        """
        Process current frame tracked vehicles and return any newly detected incidents.
        """
        now = current_time or time.time()
        new_alerts: List[IncidentAlert] = []

        # Clean expired cooldown entries
        self._prune_cooldowns(now)

        # 1. Sudden Deceleration & Stalled Vehicle Check
        for v in vehicles:
            if v.track_id not in self.recent_speeds:
                self.recent_speeds[v.track_id] = []

            speeds = self.recent_speeds[v.track_id]
            speeds.append(v.speed_kmh)
            if len(speeds) > 15:
                speeds.pop(0)

            # Check sudden sharp deceleration: speed drop > threshold into near standstill
            if len(speeds) >= 8:
                delta_v = speeds[0] - speeds[-1]
                if delta_v > self.decel_thresh and speeds[-1] < 10.0:
                    alert_key = (v.track_id, 0)
                    if alert_key not in self.reported_collision_pairs:
                        alert = IncidentAlert(
                            incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                            incident_type=IncidentType.SUDDEN_DECELERATION,
                            severity="WARNING",
                            timestamp=now,
                            track_id=v.track_id,
                            approach_id=v.approach_id,
                            location=v.bbox.center,
                            description=f"Sudden sharp deceleration on {v.class_name} #{v.track_id} ({delta_v:.1f} km/h drop)",
                            details={
                                "initial_speed_kmh": round(speeds[0], 1),
                                "final_speed_kmh": round(speeds[-1], 1),
                                "speed_drop_kmh": round(delta_v, 1),
                                "class_name": v.class_name,
                            },
                        )
                        new_alerts.append(alert)
                        self.reported_collision_pairs[alert_key] = now

            # Stalled vehicle check (stopped on active carriageway outside queued traffic)
            if v.speed_kmh < 1.5 and not v.in_queue:
                v.stopped_frames += 1
                if v.stopped_frames >= self.stalled_frames_thresh:
                    if v.track_id not in self.reported_stalls:
                        alert = IncidentAlert(
                            incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                            incident_type=IncidentType.STALLED_VEHICLE,
                            severity="CRITICAL",
                            timestamp=now,
                            track_id=v.track_id,
                            approach_id=v.approach_id,
                            location=v.bbox.center,
                            description=f"Stalled vehicle #{v.track_id} ({v.class_name}) blocking lane {v.lane_id or 'approach'}",
                            details={
                                "stopped_duration_sec": round(v.stopped_frames / self.fps, 1),
                                "lane_id": v.lane_id,
                                "class_name": v.class_name,
                            },
                        )
                        new_alerts.append(alert)
                        self.reported_stalls[v.track_id] = now
            else:
                v.stopped_frames = 0
                if v.track_id in self.reported_stalls:
                    del self.reported_stalls[v.track_id]

        # 2. Collision Check: Overlapping bounding boxes between interacting vehicles
        n = len(vehicles)
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = vehicles[i], vehicles[j]
                pair_key = (min(v1.track_id, v2.track_id), max(v1.track_id, v2.track_id))

                if pair_key in self.reported_collision_pairs:
                    continue

                if self._check_collision_overlap(v1, v2):
                    alert = IncidentAlert(
                        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.COLLISION,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v1.track_id,
                        approach_id=v1.approach_id or v2.approach_id,
                        location=(
                            (v1.bbox.center[0] + v2.bbox.center[0]) / 2.0,
                            (v1.bbox.center[1] + v2.bbox.center[1]) / 2.0,
                        ),
                        description=f"Potential collision between Vehicle #{v1.track_id} ({v1.class_name}) and #{v2.track_id} ({v2.class_name})",
                        details={
                            "vehicle_1_id": v1.track_id,
                            "vehicle_1_class": v1.class_name,
                            "vehicle_1_speed": v1.speed_kmh,
                            "vehicle_2_id": v2.track_id,
                            "vehicle_2_class": v2.class_name,
                            "vehicle_2_speed": v2.speed_kmh,
                        },
                    )
                    new_alerts.append(alert)
                    self.reported_collision_pairs[pair_key] = now

        # 3. Road Blockage Check (Multiple stalled vehicles clustering on same approach)
        stalled_on_approach: Dict[str, List[int]] = {}
        for v in vehicles:
            if v.stopped_frames >= self.stalled_frames_thresh and v.approach_id:
                stalled_on_approach.setdefault(v.approach_id, []).append(v.track_id)

        for approach_id, stalled_ids in stalled_on_approach.items():
            if len(stalled_ids) >= 2:
                block_key = (-1, hash(approach_id) % 10000)
                if block_key not in self.reported_collision_pairs:
                    alert = IncidentAlert(
                        incident_id=f"INC-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.ROAD_BLOCKAGE,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=stalled_ids[0],
                        approach_id=approach_id,
                        location=(640.0, 360.0),
                        description=f"Severe Road Blockage on {approach_id}: {len(stalled_ids)} vehicles stalled simultaneously",
                        details={"approach_id": approach_id, "stalled_vehicle_count": len(stalled_ids)},
                    )
                    new_alerts.append(alert)
                    self.reported_collision_pairs[block_key] = now

        self.active_alerts.extend(new_alerts)
        if len(self.active_alerts) > 100:
            self.active_alerts = self.active_alerts[-100:]

        return new_alerts

    def _check_collision_overlap(self, v1: TrackedVehicle, v2: TrackedVehicle) -> bool:
        """
        Check if two vehicles share bounding box overlap with abrupt impact characteristics.
        """
        x1 = max(v1.bbox.x1, v2.bbox.x1)
        y1 = max(v1.bbox.y1, v2.bbox.y1)
        x2 = min(v1.bbox.x2, v2.bbox.x2)
        y2 = min(v1.bbox.y2, v2.bbox.y2)

        if x2 > x1 and y2 > y1:
            inter_area = (x2 - x1) * (y2 - y1)
            min_area = min(v1.bbox.area, v2.bbox.area)
            if min_area > 0 and (inter_area / min_area) >= self.collision_iou_thresh:
                # High overlap with abrupt standstill or low speed (< 5 km/h) indicates collision
                if v1.speed_kmh < 5.0 or v2.speed_kmh < 5.0:
                    return True
        return False

    def _prune_cooldowns(self, now: float) -> None:
        """Prune expired cooldown timers."""
        expired = [k for k, ts in self.reported_collision_pairs.items() if now - ts > self.cooldown_sec]
        for k in expired:
            del self.reported_collision_pairs[k]

    def reset(self) -> None:
        """Reset internal detector state."""
        self.recent_speeds.clear()
        self.active_alerts.clear()
        self.reported_collision_pairs.clear()
        self.reported_stalls.clear()
