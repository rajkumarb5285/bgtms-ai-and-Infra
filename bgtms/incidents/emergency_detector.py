"""
BGTMS - Emergency Vehicle Detection & Green Corridor Preemption (M1-21)
Identifies ambulances, fire trucks, and police vehicles, initiating automated green corridor preemption.
"""

import time
import uuid
from typing import List, Tuple, Optional
from bgtms.core.types import TrackedVehicle, IncidentAlert, IncidentType


class EmergencyDetector:
    """
    Emergency vehicle detection and green corridor corridor preemption manager.
    """

    def __init__(self, stop_line_y: float = 880.0):
        self.stop_line_y = stop_line_y
        self.active_emergency_tracks: set = set()

    def process_emergency_vehicles(self, vehicles: List[TrackedVehicle], approach_id: str = "Approach_North",
                                   current_time: Optional[float] = None) -> Tuple[List[IncidentAlert], bool]:
        now = current_time or time.time()
        alerts: List[IncidentAlert] = []
        green_corridor_required = False

        for v in vehicles:
            # Check if class is emergency_vehicle (class_id == 7) or marked emergency
            if v.class_id == 7 or v.is_emergency or "ambulance" in v.class_name.lower():
                v.is_emergency = True
                green_corridor_required = True

                if v.track_id not in self.active_emergency_tracks:
                    self.active_emergency_tracks.add(v.track_id)
                    
                    dist_to_stop = max(0.0, (self.stop_line_y - v.bbox.y2) * 0.05) # approx meters
                    eta_sec = (dist_to_stop / max(5.0, v.speed_kmh / 3.6)) if v.speed_kmh > 2.0 else 15.0

                    alert = IncidentAlert(
                        incident_id=f"EMG-{uuid.uuid4().hex[:6].upper()}",
                        incident_type=IncidentType.EMERGENCY_VEHICLE_APPROACH,
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=v.track_id,
                        approach_id=approach_id,
                        location=v.bbox.center,
                        description=f"EMERGENCY VEHICLE INBOUND: {v.class_name} #{v.track_id}. Triggering Green Corridor preemption!",
                        details={
                            "speed_kmh": v.speed_kmh,
                            "estimated_arrival_sec": round(eta_sec, 1),
                            "action": "TRIGGER_GREEN_CORRIDOR",
                            "approach_id": approach_id
                        }
                    )
                    alerts.append(alert)

        return alerts, green_corridor_required
