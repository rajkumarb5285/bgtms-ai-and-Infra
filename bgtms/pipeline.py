"""
BGTMS - Unified End-to-End Traffic Intelligence Pipeline
Integrates Detection, Tracking, Speed, Lanes, Density, Queues,
Incidents, Predictions, and Signal Optimization into a single cycle.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from bgtms.core.types import (
    BoundingBox, TrackedVehicle, ApproachState, IncidentAlert, SignalPlan
)
from bgtms.vision.detector import VehicleDetector
from bgtms.vision.tracker import MultiObjectTracker
from bgtms.vision.speed_estimator import SpeedEstimator
from bgtms.vision.lane_detector import LaneDetector
from bgtms.traffic.density_engine import TrafficDensityEngine
from bgtms.traffic.queue_detector import QueueDetector
from bgtms.traffic.congestion_classifier import CongestionClassifier
from bgtms.prediction.congestion_predictor import CongestionPredictor
from bgtms.incidents.accident_detector import AccidentDetector
from bgtms.incidents.violation_detector import ViolationDetector
from bgtms.incidents.emergency_detector import EmergencyDetector
from bgtms.signal.webster_optimizer import WebsterSignalOptimizer
from bgtms.signal.sumo_runner import SumoSimulator


class TrafficIntelligencePipeline:
    """
    End-to-end multi-modal processing pipeline.
    """

    def __init__(self, approach_id: str = "North_HosurRd", approach_name: str = "Silk Board North"):
        self.approach_id = approach_id
        self.approach_name = approach_name

        # Vision submodules
        self.detector = VehicleDetector()
        self.tracker = MultiObjectTracker()
        self.speed_estimator = SpeedEstimator()
        self.lane_detector = LaneDetector()

        # Traffic & Prediction submodules
        self.density_engine = TrafficDensityEngine(approach_id=self.approach_id)
        self.queue_detector = QueueDetector(speed_estimator=self.speed_estimator)
        self.congestion_classifier = CongestionClassifier()
        self.predictor = CongestionPredictor()

        # Incident submodules
        self.accident_detector = AccidentDetector()
        self.violation_detector = ViolationDetector()
        self.emergency_detector = EmergencyDetector()

        # Signal submodules
        self.signal_optimizer = WebsterSignalOptimizer()
        self.sumo_simulator = SumoSimulator()

        # State storage
        self.latest_signal_plan = self.signal_optimizer.compute_optimal_plan({})
        self.latest_approach_state = ApproachState(
            approach_id=self.approach_id,
            approach_name=self.approach_name
        )
        self.all_incidents: List[IncidentAlert] = []
        self.frame_index = 0

    def process_frame(
        self,
        frame: np.ndarray,
        is_red_light: bool = False,
        conf_threshold: Optional[float] = None,
        is_single_frame: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a single video frame end-to-end.
        """
        start_time = time.time()
        self.frame_index += 1
        fh, fw = frame.shape[:2]

        # 1. Detection
        detections: List[BoundingBox] = self.detector.detect(frame, conf_threshold=conf_threshold)

        # 2. Tracking
        tracked_vehicles: List[TrackedVehicle] = self.tracker.update(detections, is_single_frame=is_single_frame)

        # 3. Speed & Lane Association
        for veh in tracked_vehicles:
            veh.approach_id = self.approach_id
            self.speed_estimator.estimate_speed(veh)
            self.lane_detector.associate_vehicle_to_lane(veh, frame_w=fw, frame_h=fh)


        # 4. Traffic Density & Queues
        lane_metrics = self.density_engine.compute_lane_metrics(tracked_vehicles, num_lanes=3)
        for lm in lane_metrics:
            q_m, waiting_cnt, growth = self.queue_detector.compute_queue(lm.lane_id, tracked_vehicles)
            lm.queue_length_m = q_m
            lm.waiting_vehicles = waiting_cnt

        # 5. Congestion Classification
        approach_state = self.congestion_classifier.classify(
            lanes=lane_metrics,
            approach_id=self.approach_id,
            approach_name=self.approach_name
        )
        self.latest_approach_state = approach_state
        self.predictor.record_step(approach_state)

        # 6. Incident Analysis
        accidents = self.accident_detector.check_incidents(tracked_vehicles, current_time=start_time)
        violations = self.violation_detector.check_violations(tracked_vehicles, is_red_light=is_red_light, current_time=start_time)
        emergencies, green_corridor_req = self.emergency_detector.process_emergency_vehicles(
            tracked_vehicles, approach_id=self.approach_id, current_time=start_time
        )

        new_incidents = accidents + violations + emergencies
        self.all_incidents.extend(new_incidents)

        # 7. Adaptive Signal Optimization
        approach_map = {self.approach_id: approach_state}
        green_corridor_target = self.approach_id if green_corridor_req else None
        signal_plan = self.signal_optimizer.compute_optimal_plan(
            approaches=approach_map,
            green_corridor_approach=green_corridor_target
        )
        self.latest_signal_plan = signal_plan

        # 8. SUMO Closed-Loop Step
        sim_step = self.sumo_simulator.step_simulation(signal_plan, dt_sec=0.033)

        # 9. Predictions
        forecasts = self.predictor.predict()

        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "frame_index": self.frame_index,
            "latency_ms": round(elapsed_ms, 2),
            "fps": round(1000.0 / max(1.0, elapsed_ms), 1),
            "detections_count": len(detections),
            "tracked_vehicles": [v.model_dump() for v in tracked_vehicles],
            "approach_state": approach_state.model_dump(),
            "signal_plan": signal_plan.model_dump(),
            "new_incidents": [inc.model_dump() for inc in new_incidents],
            "total_incidents_count": len(self.all_incidents),
            "forecasts": forecasts,
            "simulation": sim_step
        }

    def render_annotated_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Produces fully annotated visual frame with lanes, bounding boxes,
        speeds, telemetry HUD, and alert banners.
        """
        vis = frame.copy()
        
        # Draw lanes
        vis = self.lane_detector.draw_lanes(vis, alpha=0.18)

        # Draw tracked vehicles
        for v in result.get("tracked_vehicles", []):
            x1, y1 = int(v["bbox"]["x1"]), int(v["bbox"]["y1"])
            x2, y2 = int(v["bbox"]["x2"]), int(v["bbox"]["y2"])
            cls_name = v["class_name"]
            track_id = v["track_id"]
            spd = v["speed_kmh"]
            is_emg = v["is_emergency"]

            box_color = (0, 0, 255) if is_emg else (0, 200, 255) if v["in_queue"] else (0, 255, 120)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

            label = f"#{track_id} {cls_name} {spd:.0f}km/h"
            if is_emg:
                label += " [EMERGENCY]"
            elif v["in_queue"]:
                label += " [QUEUE]"

            cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD Top Banner
        app_state = result.get("approach_state", {})
        sig = result.get("signal_plan", {})
        fps = result.get("fps", 30.0)
        cong = app_state.get("congestion_level", "LOW")

        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 50), (18, 22, 30), -1)
        
        hud_text = f"BGTMS | {self.approach_name} | FPS: {fps:.1f} | PCU: {app_state.get('total_pcu', 0)} | Queue: {app_state.get('queue_length_m', 0)}m | Congestion: {cong}"
        cv2.putText(vis, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA)

        # Green corridor alert banner if active
        if sig.get("is_green_corridor"):
            cv2.rectangle(vis, (0, h - 50), (w, h), (0, 0, 200), -1)
            cv2.putText(vis, ">>> GREEN CORRIDOR ACTIVE - EMERGENCY PREEMPTION <<<", (w // 2 - 300, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return vis
