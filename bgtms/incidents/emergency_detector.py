"""
BGTMS - Emergency Vehicle Detection & Green Corridor Preemption (M1-21 / M2)
Identifies ambulances, fire trucks, and police vehicles, initiating automated green corridor preemption.
"""

import time
import uuid
from typing import List, Tuple, Optional, Dict, Set
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType, SignalPlan, SignalPhase


class EmergencyDetector:
    """
    Emergency vehicle detection, trajectory tracking, and automated green corridor preemption manager.
    """

    def __init__(self, stop_line_y: float = 880.0, exit_line_y: float = 950.0):
        self.stop_line_y = stop_line_y
        self.exit_line_y = exit_line_y
        self.active_emergency_tracks: Set[int] = set()
        self.cleared_emergency_tracks: Set[int] = set()
        self.corridor_in_progress: bool = False
        self.active_corridor_approach: Optional[str] = None
        self.corridor_start_time: Optional[float] = None

    def process_emergency_vehicles(
        self,
        vehicles: List[TrackedVehicle],
        approach_id: str = "Approach_North",
        current_time: Optional[float] = None,
    ) -> Tuple[List[IncidentAlert], bool]:
        """
        Scan tracked vehicles for emergency vehicles and determine green corridor requirement.
        """
        now = current_time or time.time()
        alerts: List[IncidentAlert] = []
        green_corridor_required = False

        for v in vehicles:
            is_emergency = (
                v.class_id == 7
                or v.is_emergency
                or "ambulance" in v.class_name.lower()
                or "emergency" in v.class_name.lower()
                or "fire" in v.class_name.lower()
            )

            if is_emergency:
                v.is_emergency = True

                # Check if already exited intersection
                if v.bbox.y2 > self.exit_line_y:
                    if v.track_id in self.active_emergency_tracks:
                        self.active_emergency_tracks.remove(v.track_id)
                        self.cleared_emergency_tracks.add(v.track_id)
                    continue

                green_corridor_required = True
                self.corridor_in_progress = True
                self.active_corridor_approach = approach_id

                if v.track_id not in self.active_emergency_tracks and v.track_id not in self.cleared_emergency_tracks:
                    self.active_emergency_tracks.add(v.track_id)
                    self.corridor_start_time = now

                    dist_to_stop = max(0.0, (self.stop_line_y - v.bbox.y2) * 0.05)  # approx meters
                    eta_sec = (dist_to_stop / max(3.0, v.speed_kmh / 3.6)) if v.speed_kmh > 1.0 else 15.0

                    alert = IncidentAlert(
                        incident_id=f"EMG-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.EMERGENCY_VEHICLE_APPROACH,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=approach_id,
                        location=v.bbox.center,
                        description=f"EMERGENCY VEHICLE INBOUND: {v.class_name} #{v.track_id}. Activating Green Corridor Preemption!",
                        details={
                            "class_name": v.class_name,
                            "speed_kmh": round(v.speed_kmh, 1),
                            "estimated_arrival_sec": round(eta_sec, 1),
                            "action": "TRIGGER_GREEN_CORRIDOR",
                            "corridor_approach": approach_id,
                        },
                    )
                    alerts.append(alert)

        # If no active emergency vehicles remain, end corridor
        if not self.active_emergency_tracks:
            self.corridor_in_progress = False
            self.active_corridor_approach = None

        return alerts, green_corridor_required

    def build_preemption_signal_plan(
        self, junction_id: str, priority_approach: str, min_green_sec: float = 30.0
    ) -> SignalPlan:
        """
        Construct a preemptive green corridor signal plan holding conflicting arms in all-red.
        """
        # Determine priority phase (Phase 1 for North/South, Phase 2 for East/West)
        is_ns = "north" in priority_approach.lower() or "south" in priority_approach.lower()
        priority_phase_id = 1 if is_ns else 2

        phases = [
            SignalPhase(
                phase_id=1,
                name="North-South Green Corridor",
                green_duration_sec=min_green_sec if is_ns else 5.0,
                yellow_duration_sec=3.0,
                all_red_duration_sec=2.0,
                allocated_approaches=["Approach_North", "Approach_South"],
            ),
            SignalPhase(
                phase_id=2,
                name="East-West Green Corridor",
                green_duration_sec=5.0 if is_ns else min_green_sec,
                yellow_duration_sec=3.0,
                all_red_duration_sec=2.0,
                allocated_approaches=["Approach_East", "Approach_West"],
            ),
        ]

        return SignalPlan(
            junction_id=junction_id,
            cycle_time_sec=min_green_sec + 10.0,
            current_phase_id=priority_phase_id,
            time_in_phase_sec=0.0,
            phases=phases,
            is_green_corridor=True,
            green_corridor_approach=priority_approach,
        )

    def reset(self) -> None:
        """Reset internal emergency state."""
        self.active_emergency_tracks.clear()
        self.cleared_emergency_tracks.clear()
        self.corridor_in_progress = False
        self.active_corridor_approach = None
        self.corridor_start_time = None
