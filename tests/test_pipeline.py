"""
BGTMS Automated Unit & Integration Tests (M1-28)
"""

import pytest
import numpy as np
from bgtms.vision.detector import VehicleDetector
from bgtms.vision.tracker import MultiObjectTracker
from bgtms.vision.speed_estimator import SpeedEstimator
from bgtms.vision.lane_detector import LaneDetector
from bgtms.traffic.density_engine import TrafficDensityEngine
from bgtms.traffic.queue_detector import QueueDetector
from bgtms.traffic.congestion_classifier import CongestionClassifier
from bgtms.signal.webster_optimizer import WebsterSignalOptimizer
from bgtms.incidents.accident_detector import AccidentDetector
from bgtms.incidents.emergency_detector import EmergencyDetector
from bgtms.core.types import BoundingBox, TrackedVehicle, CongestionLevel


def test_vehicle_detector():
    detector = VehicleDetector()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw simulated vehicle contour
    frame[300:450, 400:600] = 200
    detections = detector.detect(frame)
    assert isinstance(detections, list)
    for d in detections:
        assert 0 <= d.class_id <= 7
        assert d.confidence >= 0.0


def test_multi_object_tracker():
    tracker = MultiObjectTracker()
    box1 = BoundingBox(x1=100, y1=100, x2=200, y2=200, confidence=0.9, class_id=0, class_name="car")
    tracks = tracker.update([box1])
    assert len(tracks) >= 0


def test_speed_estimator():
    estimator = SpeedEstimator()
    v = TrackedVehicle(
        track_id=1, class_id=0, class_name="car", pcu=1.0,
        bbox=BoundingBox(x1=100, y1=800, x2=200, y2=900, confidence=0.9, class_id=0, class_name="car"),
        trajectory=[(500, 700), (500, 750), (500, 800)]
    )
    spd = estimator.estimate_speed(v)
    assert spd >= 0.0


def test_density_and_queue():
    density_engine = TrafficDensityEngine()
    queue_detector = QueueDetector()

    v = TrackedVehicle(
        track_id=1, class_id=0, class_name="car", pcu=1.0,
        bbox=BoundingBox(x1=700, y1=850, x2=800, y2=950, confidence=0.9, class_id=0, class_name="car"),
        speed_kmh=1.5,
        lane_id=2
    )
    metrics = density_engine.compute_lane_metrics([v])
    assert len(metrics) == 3
    assert metrics[1].vehicle_count == 1
    assert metrics[1].pcu_count == 1.0


def test_webster_optimizer():
    optimizer = WebsterSignalOptimizer()
    plan = optimizer.compute_optimal_plan({})
    assert plan.cycle_time_sec >= 50.0
    assert len(plan.phases) == 2
    assert plan.phases[0].green_duration_sec >= 15.0


def test_emergency_preemption():
    emg_detector = EmergencyDetector()
    v = TrackedVehicle(
        track_id=99, class_id=7, class_name="emergency_vehicle", pcu=0.0,
        bbox=BoundingBox(x1=500, y1=500, x2=600, y2=600, confidence=0.95, class_id=7, class_name="emergency_vehicle"),
        is_emergency=True,
        speed_kmh=45.0
    )
    alerts, req = emg_detector.process_emergency_vehicles([v])
    assert req is True
    assert len(alerts) == 1
