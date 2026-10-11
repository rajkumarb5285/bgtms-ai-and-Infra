"""
BGTMS Core Data Types and Schemas
"""

from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class CongestionLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentType(str, Enum):
    COLLISION = "COLLISION"
    SUDDEN_DECELERATION = "SUDDEN_DECELERATION"
    STALLED_VEHICLE = "STALLED_VEHICLE"
    ROAD_BLOCKAGE = "ROAD_BLOCKAGE"
    RED_LIGHT_VIOLATION = "RED_LIGHT_VIOLATION"
    WRONG_WAY = "WRONG_WAY"
    STOP_LINE_VIOLATION = "STOP_LINE_VIOLATION"
    EMERGENCY_VEHICLE_APPROACH = "EMERGENCY_VEHICLE_APPROACH"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class TrackedVehicle(BaseModel):
    track_id: int
    class_id: int
    class_name: str
    pcu: float
    bbox: BoundingBox
    trajectory: List[Tuple[float, float]] = Field(default_factory=list)
    speed_kmh: float = 0.0
    lane_id: Optional[int] = None
    approach_id: Optional[str] = None
    is_emergency: bool = False
    stopped_frames: int = 0
    in_queue: bool = False


class LaneMetrics(BaseModel):
    lane_id: int
    approach_id: str
    vehicle_count: int = 0
    pcu_count: float = 0.0
    density_veh_km: float = 0.0
    occupancy_pct: float = 0.0
    flow_pcu_hr: float = 0.0
    avg_speed_kmh: float = 0.0
    queue_length_m: float = 0.0
    waiting_vehicles: int = 0


class ApproachState(BaseModel):
    approach_id: str
    approach_name: str
    total_vehicles: int = 0
    total_pcu: float = 0.0
    average_speed_kmh: float = 0.0
    queue_length_m: float = 0.0
    congestion_level: CongestionLevel = CongestionLevel.LOW
    lanes: List[LaneMetrics] = Field(default_factory=list)


class IncidentAlert(BaseModel):
    incident_id: str
    incident_type: IncidentType
    severity: str  # INFO, WARNING, CRITICAL
    timestamp: float
    track_id: Optional[int] = None
    approach_id: Optional[str] = None
    location: Tuple[float, float]
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class SignalPhase(BaseModel):
    phase_id: int
    name: str
    green_duration_sec: float
    yellow_duration_sec: float = 4.0
    all_red_duration_sec: float = 2.0
    allocated_approaches: List[str]


class SignalPlan(BaseModel):
    junction_id: str
    cycle_time_sec: float
    current_phase_id: int
    time_in_phase_sec: float
    phases: List[SignalPhase]
    is_green_corridor: bool = False
    green_corridor_approach: Optional[str] = None
