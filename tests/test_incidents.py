"""
Unit & Integration Tests for BGTMS Incident & Safety Intelligence Engine (M2)
Tests AccidentDetector, EmergencyDetector, and ViolationDetector.
"""

import pytest
import time
from bgtms.core.types import TrackedVehicle, BoundingBox, IncidentType, CongestionLevel
from bgtms.incidents.accident_detector import AccidentDetector
from bgtms.incidents.emergency_detector import EmergencyDetector
from bgtms.incidents.violation_detector import ViolationDetector


def make_vehicle(
    track_id: int,
    class_id: int = 0,
    class_name: str = "car",
    pcu: float = 1.0,
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 160.0,
    y2: float = 180.0,
    speed_kmh: float = 30.0,
    stopped_frames: int = 0,
    in_queue: bool = False,
    is_emergency: bool = False,
    approach_id: str = "Approach_North",
    lane_id: int = 1,
) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        pcu=pcu,
        bbox=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=0.92,
            class_id=class_id,
            class_name=class_name,
        ),
        trajectory=[(x1 + 30.0, y1 + 40.0)],
        speed_kmh=speed_kmh,
        lane_id=lane_id,
        approach_id=approach_id,
        is_emergency=is_emergency,
        stopped_frames=stopped_frames,
        in_queue=in_queue,
    )


class TestAccidentDetector:
    def test_sudden_deceleration_detection(self):
        detector = AccidentDetector(sudden_decel_thresh=20.0, fps=10.0)
        v = make_vehicle(track_id=101, speed_kmh=45.0)

        # Feed decreasing speeds
        speeds = [45.0, 42.0, 40.0, 35.0, 25.0, 15.0, 8.0, 3.0]
        alerts = []
        for s in speeds:
            v.speed_kmh = s
            alerts = detector.check_incidents([v])

        assert any(a.incident_type == IncidentType.SUDDEN_DECELERATION for a in alerts)
        decel_alert = [a for a in alerts if a.incident_type == IncidentType.SUDDEN_DECELERATION][0]
        assert decel_alert.track_id == 101
        assert decel_alert.severity == "WARNING"

    def test_collision_detection(self):
        detector = AccidentDetector(collision_iou_thresh=0.35)
        # Two vehicles colliding with high box overlap and low speed post-impact
        v1 = make_vehicle(track_id=201, class_name="car", x1=200, y1=200, x2=280, y2=280, speed_kmh=2.0)
        v2 = make_vehicle(track_id=202, class_name="motorcycle", x1=210, y1=210, x2=270, y2=270, speed_kmh=0.0)

        alerts = detector.check_incidents([v1, v2])
        collisions = [a for a in alerts if a.incident_type == IncidentType.COLLISION]
        assert len(collisions) >= 1
        assert collisions[0].severity == "CRITICAL"
        assert "201" in collisions[0].description
        assert "202" in collisions[0].description

    def test_stalled_vehicle_detection(self):
        fps = 10.0
        stalled_time_sec = 2.0  # 20 frames
        detector = AccidentDetector(stalled_time_sec=stalled_time_sec, fps=fps)

        v = make_vehicle(track_id=301, speed_kmh=0.5, stopped_frames=0, in_queue=False)
        alerts = []
        for _ in range(25):
            alerts.extend(detector.check_incidents([v]))

        stall_alerts = [a for a in alerts if a.incident_type == IncidentType.STALLED_VEHICLE]
        assert len(stall_alerts) >= 1
        assert stall_alerts[0].track_id == 301
        assert stall_alerts[0].severity == "CRITICAL"


class TestEmergencyDetector:
    def test_emergency_vehicle_detection_and_preemption(self):
        detector = EmergencyDetector(stop_line_y=880.0, exit_line_y=950.0)
        ambulance = make_vehicle(
            track_id=401,
            class_id=7,
            class_name="emergency_vehicle",
            y1=700,
            y2=780,
            speed_kmh=40.0,
            approach_id="Approach_North",
        )

        alerts, req_preemption = detector.process_emergency_vehicles([ambulance], approach_id="Approach_North")
        assert req_preemption is True
        assert len(alerts) == 1
        assert alerts[0].incident_type == IncidentType.EMERGENCY_VEHICLE_APPROACH
        assert alerts[0].details["action"] == "TRIGGER_GREEN_CORRIDOR"

        # Build signal plan
        plan = detector.build_preemption_signal_plan("SilkBoard_Junction", "Approach_North", min_green_sec=35.0)
        assert plan.is_green_corridor is True
        assert plan.green_corridor_approach == "Approach_North"
        assert plan.current_phase_id == 1
        assert plan.phases[0].green_duration_sec == 35.0

    def test_emergency_corridor_clearance(self):
        detector = EmergencyDetector(stop_line_y=880.0, exit_line_y=950.0)
        ambulance = make_vehicle(
            track_id=402,
            class_name="ambulance",
            y1=700,
            y2=780,
            approach_id="Approach_North",
        )
        detector.process_emergency_vehicles([ambulance])
        assert detector.corridor_in_progress is True

        # Vehicle passes exit line (y2 > 950)
        ambulance.bbox.y1 = 920
        ambulance.bbox.y2 = 980
        _, req_preemption = detector.process_emergency_vehicles([ambulance])
        assert req_preemption is False
        assert detector.corridor_in_progress is False


class TestViolationDetector:
    def test_red_light_running_violation(self):
        detector = ViolationDetector(stop_line_y=800.0, encroach_buffer_px=30.0)
        v = make_vehicle(track_id=501, speed_kmh=35.0, y1=810, y2=860)
        # Previous trajectory point was behind stop line (y <= 800)
        v.trajectory = [(150, 780), (150, 840)]

        violations = detector.check_violations([v], is_red_light=True)
        assert len(violations) >= 1
        assert violations[0].incident_type == IncidentType.RED_LIGHT_VIOLATION
        assert violations[0].severity == "CRITICAL"

    def test_stop_line_encroachment(self):
        detector = ViolationDetector(stop_line_y=800.0, encroach_buffer_px=40.0)
        # Vehicle stopping over the line within encroachment buffer (800 < y2 <= 840)
        v = make_vehicle(track_id=502, speed_kmh=1.0, y1=770, y2=820)
        v.trajectory = [(150, 810)]

        violations = detector.check_violations([v], is_red_light=True)
        assert len(violations) >= 1
        assert violations[0].incident_type == IncidentType.STOP_LINE_VIOLATION
        assert violations[0].severity == "WARNING"

    def test_wrong_way_violation(self):
        detector = ViolationDetector()
        v = make_vehicle(track_id=503, speed_kmh=25.0)
        # Inverted trajectory (y decreasing sharply from 500 to 420)
        v.trajectory = [
            (200, 500), (200, 485), (200, 470),
            (200, 455), (200, 440), (200, 420)
        ]

        violations = detector.check_violations([v], is_red_light=False)
        assert len(violations) >= 1
        assert violations[0].incident_type == IncidentType.WRONG_WAY
        assert violations[0].severity == "CRITICAL"
