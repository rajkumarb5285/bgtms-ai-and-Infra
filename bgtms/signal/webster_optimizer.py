"""
BGTMS - Webster Adaptive Signal Optimization Engine (M1-23)
Calculates optimal cycle time and phase allocations using Webster's minimum-delay formulation.
"""

from typing import Dict, List, Optional
from bgtms.core.types import SignalPlan, SignalPhase, ApproachState


class WebsterSignalOptimizer:
    """
    Adaptive Traffic Light Timing Optimizer based on Webster's Method:
        C_opt = (1.5 * L + 5) / (1 - Y)
        g_i = (C_opt - L) * (y_i / Y)
    """

    def __init__(self, saturation_flow_pcu_per_lane: float = 1800.0,
                 lost_time_per_phase_sec: float = 4.0,
                 min_green_sec: float = 15.0,
                 max_green_sec: float = 75.0,
                 yellow_sec: float = 4.0,
                 all_red_sec: float = 2.0):
        self.sat_flow = saturation_flow_pcu_per_lane
        self.lost_time_per_phase = lost_time_per_phase_sec
        self.min_green = min_green_sec
        self.max_green = max_green_sec
        self.yellow = yellow_sec
        self.all_red = all_red_sec

    def compute_optimal_plan(self, approaches: Dict[str, ApproachState],
                             green_corridor_approach: Optional[str] = None) -> SignalPlan:
        """
        Calculates optimized phase timing based on 4 approaches:
        Phase 1: North-South (Hosur Rd / Electronic City corridor)
        Phase 2: East-West (HSR Layout / BTM corridor)
        """
        # If green corridor requested, give immediate dedicated phase
        if green_corridor_approach:
            return SignalPlan(
                junction_id="BLR-SILKBOARD-01",
                cycle_time_sec=90.0,
                current_phase_id=1 if "North" in green_corridor_approach or "South" in green_corridor_approach else 2,
                time_in_phase_sec=0.0,
                is_green_corridor=True,
                green_corridor_approach=green_corridor_approach,
                phases=[
                    SignalPhase(
                        phase_id=1,
                        name="North-South Green Corridor (EMERGENCY OVERRIDE)",
                        green_duration_sec=65.0,
                        yellow_duration_sec=3.0,
                        all_red_duration_sec=2.0,
                        allocated_approaches=["North_HosurRd", "South_ElectronicCity"]
                    ),
                    SignalPhase(
                        phase_id=2,
                        name="East-West Holding Phase",
                        green_duration_sec=20.0,
                        yellow_duration_sec=3.0,
                        all_red_duration_sec=2.0,
                        allocated_approaches=["East_HSRLayout", "West_BTM"]
                    )
                ]
            )

        # Normal Webster computation
        # Group 1: North & South critical flow
        flow_n = approaches.get("North_HosurRd", ApproachState(approach_id="North", approach_name="North")).total_pcu
        flow_s = approaches.get("South_ElectronicCity", ApproachState(approach_id="South", approach_name="South")).total_pcu
        crit_q1 = max(flow_n, flow_s)

        # Group 2: East & West critical flow
        flow_e = approaches.get("East_HSRLayout", ApproachState(approach_id="East", approach_name="East")).total_pcu
        crit_q2 = max(flow_e, flow_w := approaches.get("West_BTM", ApproachState(approach_id="West", approach_name="West")).total_pcu)

        # Scale hourly flow rate
        q1_hr = max(150.0, crit_q1 * 60.0) # Est hourly demand
        q2_hr = max(100.0, crit_q2 * 60.0)

        # Saturation flow for 3 lanes
        sat_total = self.sat_flow * 3.0

        y1 = min(0.85, q1_hr / sat_total)
        y2 = min(0.85, q2_hr / sat_total)
        Y = y1 + y2

        # Lost time: L = 2 phases * lost_time_per_phase
        L = 2.0 * self.lost_time_per_phase

        if Y >= 0.90:
            # Saturated condition, use max stable cycle length
            C_opt = 120.0
        else:
            C_opt = (1.5 * L + 5.0) / (1.0 - Y)
            C_opt = max(50.0, min(140.0, C_opt))

        # Effective green time
        G_total = max(20.0, C_opt - L)
        g1 = G_total * (y1 / max(1e-4, Y))
        g2 = G_total * (y2 / max(1e-4, Y))

        # Clamp green durations
        g1 = max(self.min_green, min(self.max_green, g1))
        g2 = max(self.min_green, min(self.max_green, g2))
        cycle_time = round(g1 + g2 + (2.0 * (self.yellow + self.all_red)), 1)

        return SignalPlan(
            junction_id="BLR-SILKBOARD-01",
            cycle_time_sec=cycle_time,
            current_phase_id=1,
            time_in_phase_sec=0.0,
            is_green_corridor=False,
            phases=[
                SignalPhase(
                    phase_id=1,
                    name="North-South Phase (Hosur Rd / Electronic City)",
                    green_duration_sec=round(g1, 1),
                    yellow_duration_sec=self.yellow,
                    all_red_duration_sec=self.all_red,
                    allocated_approaches=["North_HosurRd", "South_ElectronicCity"]
                ),
                SignalPhase(
                    phase_id=2,
                    name="East-West Phase (HSR Layout / BTM)",
                    green_duration_sec=round(g2, 1),
                    yellow_duration_sec=self.yellow,
                    all_red_duration_sec=self.all_red,
                    allocated_approaches=["East_HSRLayout", "West_BTM"]
                )
            ]
        )
